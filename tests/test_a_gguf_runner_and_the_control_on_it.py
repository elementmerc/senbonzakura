# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The GGUF runner, and the agreement control Q-88 makes a condition on it.

TWO HALVES, AND THE SECOND IS THE ONE THAT MATTERS

The first half is the plumbing: a subprocess that may not exist, may not start, may start and
never answer, may answer with something that is not JSON, and may answer with JSON that has no
reply in it. Every one of those is a path a user can reach on a machine we will never see, so
every one of them is here with the message it produces asserted rather than just its type.

The second half is the control. A second runner for a measurement is a second instrument, and
two instruments that disagree quietly are worse than one, because every number afterwards
depends on which path produced it. So `agreement` has to refuse in each of the ways it can
refuse, and the refusal has to name which condition failed. A control that cannot fail is
decoration.

NO SERVER IS STARTED HERE. The real round trip was run by hand against the pinned llama.cpp
build and is recorded in the plan; what these tests own is every branch around it, which a real
server would exercise one of per run.
"""
import json
import subprocess
import urllib.error
import urllib.request

import pytest

from senbonzakura import ggufrun, metrics
from senbonzakura.ggufrun import GgufRunError
from tests.test_gguf_io import _gguf


# ── fakes, kept small and obvious ────────────────────────────────────────────────
class _Proc:
    """A subprocess that behaves however a test needs it to."""

    def __init__(self, *, exits=None, ignores_terminate=False):
        self.returncode = exits
        self._ignores = ignores_terminate
        self.terminated = self.killed = False
        self._waits = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self._ignores:
            self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self._waits += 1
        if self._ignores and self._waits == 1:
            raise subprocess.TimeoutExpired("llama-server", timeout)
        return self.returncode


class _Response:
    def __init__(self, payload, status=200):
        self._raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.status = status

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _serve(monkeypatch, answers, *, health=200):
    """Make urlopen answer from `answers`, keyed by the path it is asked for."""
    seen = []

    def urlopen(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url  # /health is asked for by URL
        path = "/" + url.split("/", 3)[3] if url.count("/") >= 3 else url
        seen.append(path)
        if path == "/health":
            if isinstance(health, Exception):
                raise health
            return _Response({"status": "ok"}, status=health)
        answer = answers[path]
        if isinstance(answer, Exception):
            raise answer
        return _Response(answer)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return seen


@pytest.fixture
def runner(monkeypatch, tmp_path):
    """A runner whose binary and model exist and whose process is a fake."""
    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    gguf = _gguf(tmp_path, name="toy-F16.gguf", ftype=1, arch="llama", tensors=3)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _Proc())
    monkeypatch.setattr(ggufrun, "server_identity",
                        lambda exe_, source, log=print: {"tool": "llama-server",
                                                         "source": source, "path": str(exe_)})
    return ggufrun.GgufRunner(gguf, binary=str(exe), log=lambda *_: None)


# ── what was loaded ──────────────────────────────────────────────────────────────
def test_the_model_identity_names_the_file_rather_than_the_model(tmp_path):
    ident = ggufrun.model_identity(_gguf(tmp_path, name="toy-F16.gguf", ftype=1, arch="llama"))
    assert len(ident["sha256"]) == 64, "a figure that cannot name the file it measured"
    assert ident["file_type"] == "F16"
    assert ident["architecture"] == "llama"
    assert ident["lossless"] is True
    assert ident["bytes"] > 0


@pytest.mark.parametrize(("ftype", "lossless"), [(0, True), (1, True), (32, True),
                                                 (7, False), (15, False)])
def test_lossless_is_read_from_the_file_and_not_from_the_name(tmp_path, ftype, lossless):
    ident = ggufrun.model_identity(_gguf(tmp_path, name="whatever.gguf", ftype=ftype))
    assert ident["lossless"] is lossless


def test_a_file_that_is_not_a_gguf_is_refused_before_a_server_starts(tmp_path):
    bad = tmp_path / "not.gguf"
    bad.write_bytes(b"this is not a GGUF at all")
    with pytest.raises(GgufRunError, match="not a GGUF this tool can read"):
        ggufrun.model_identity(bad)


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(GgufRunError, match="there is no file at"):
        ggufrun.model_identity(tmp_path / "absent.gguf")


def test_an_unreadable_tensor_table_leaves_the_census_absent(tmp_path, monkeypatch):
    """The census is a diagnostic. A header that parsed is enough to run."""
    from senbonzakura import gguf_io

    def boom(*_a, **_k):
        raise gguf_io.GGUFError("tensor table is past the read window")

    monkeypatch.setattr(gguf_io, "type_census", boom)
    ident = ggufrun.model_identity(_gguf(tmp_path, ftype=1))
    assert ident["tensor_types"] is None
    assert ident["file_type"] == "F16"


# ── which binary ran ─────────────────────────────────────────────────────────────
def test_the_binary_identity_carries_the_build_it_reports(tmp_path, monkeypatch):
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"#!/bin/sh\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, "version: 0.4.1-dev (build 11046, commit 60081bb2b)\n", ""))
    monkeypatch.setattr("senbonzakura.quantise.pinned_tag", lambda: "b11046")
    ident = ggufrun.server_identity(exe, "vendored")
    assert ident["reported_build"] == {"build": 11046, "commit": "60081bb2b"}
    assert ident["pinned_tag"] == "b11046"
    assert "pin_mismatch" not in ident
    assert len(ident["sha256"]) == 64


def test_a_binary_that_is_not_the_pinned_build_says_so_out_loud(tmp_path, monkeypatch):
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"#!/bin/sh\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, "version: 0.3.0 (build 9999, commit deadbeef)\n", ""))
    monkeypatch.setattr("senbonzakura.quantise.pinned_tag", lambda: "b11046")
    said = []
    ident = ggufrun.server_identity(exe, "system", log=said.append)
    assert ident["pin_mismatch"] is True
    assert any("not the one this install claims to vendor" in line for line in said)


def test_a_binary_that_will_not_say_its_build_leaves_the_field_absent(tmp_path, monkeypatch):
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"#!/bin/sh\n")

    def boom(*_a, **_k):
        raise OSError("exec format error")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr("senbonzakura.quantise.pinned_tag", lambda: None)
    ident = ggufrun.server_identity(exe, "system")
    assert ident["reported_build"] is None, "a provenance field nobody can trust"
    assert ident["pinned_tag"] is None


def test_a_version_banner_with_no_build_in_it_is_not_invented(tmp_path, monkeypatch):
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"#!/bin/sh\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, "some other program entirely\n", ""))
    monkeypatch.setattr("senbonzakura.quantise.pinned_tag", lambda: "b11046")
    assert ggufrun.server_identity(exe, "system")["reported_build"] is None


def test_a_binary_that_cannot_be_hashed_still_identifies_itself(tmp_path, monkeypatch):
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"#!/bin/sh\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, "version: 0.4.1-dev (build 11046, commit 60081bb2b)\n", ""))
    monkeypatch.setattr("senbonzakura.quantise.pinned_tag", lambda: "b11046")
    monkeypatch.setattr(ggufrun, "_sha256", lambda *_a, **_k: (_ for _ in ()).throw(OSError("gone")))
    assert ggufrun.server_identity(exe, "vendored")["sha256"] is None


# ── constructing a runner ────────────────────────────────────────────────────────
@pytest.mark.parametrize(("kwargs", "expected"), [
    ({"threads": 0}, "threads must be at least 1"),
    ({"max_new": 0}, "reply budget must be at least 1"),
])
def test_settings_that_cannot_produce_a_measurement_are_refused(tmp_path, kwargs, expected):
    gguf = _gguf(tmp_path, ftype=1)
    with pytest.raises(GgufRunError, match=expected):
        ggufrun.GgufRunner(gguf, binary="/bin/true", **kwargs)


def test_a_named_binary_that_does_not_exist_is_refused(tmp_path):
    gguf = _gguf(tmp_path, ftype=1)
    with pytest.raises(GgufRunError, match="named explicitly"):
        ggufrun.GgufRunner(gguf, binary=str(tmp_path / "nope"))


def test_with_no_binary_anywhere_the_refusal_says_why_a_full_install_lacks_one(tmp_path, monkeypatch):
    from senbonzakura import vendored

    def boom(*_a, **_k):
        raise vendored.VendorError("llama-server is not available")

    monkeypatch.setattr(vendored, "find_binary", boom)
    with pytest.raises(GgufRunError) as e:
        ggufrun.GgufRunner(_gguf(tmp_path, ftype=1), log=lambda *_: None)
    said = str(e.value)
    assert "vendor_llama.py" in said, "the refusal does not say where the binary comes from"
    assert "verifies its hash" in said, (
        "the refusal does not say the vendored route is a pinned one")


def test_a_located_binary_is_used(tmp_path, monkeypatch):
    from senbonzakura import vendored

    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setattr(vendored, "find_binary", lambda *a, **k: (str(exe), "vendored"))
    monkeypatch.setattr(ggufrun, "server_identity",
                        lambda e, s, log=print: {"source": s, "path": str(e)})
    r = ggufrun.GgufRunner(_gguf(tmp_path, ftype=1), log=lambda *_: None)
    assert r.identity["source"] == "vendored"


# ── starting and stopping ────────────────────────────────────────────────────────
def test_a_started_runner_answers_and_records_what_produced_the_answer(runner, monkeypatch):
    _serve(monkeypatch, {"/apply-template": {"prompt": "<u>hi</u>"},
                         "/completion": {"content": "I can't help with that."}})
    with runner as r:
        assert r.rendered("hi") == "<u>hi</u>"
        assert r.replies(["hi"]) == ["I can't help with that."]
        prov = r.provenance()
    assert prov["settings"]["decoding"] == "greedy"
    assert prov["settings"]["threads"] == ggufrun.DEFAULT_THREADS
    assert prov["settings"]["cache_prompt"] is False
    assert prov["model"]["file_type"] == "F16"


def test_the_sampler_chain_is_cut_to_greedy_on_every_request(runner, monkeypatch):
    """The defaults carry repeat, dry, xtc and min-p. A measurement must carry none of them."""
    sent = []

    def urlopen(req, timeout=None):
        if isinstance(req, str) or req.full_url.endswith("/health"):
            return _Response({"ok": True})
        sent.append(json.loads(req.data))
        return _Response({"content": "x", "prompt": "x"})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with runner as r:
        r.reply("rendered")
    assert sent[-1]["temperature"] == 0.0
    assert sent[-1]["top_k"] == 1
    assert sent[-1]["samplers"] == ["temperature"]
    assert sent[-1]["cache_prompt"] is False


def test_starting_twice_is_refused(runner, monkeypatch):
    _serve(monkeypatch, {})
    runner.start()
    try:
        with pytest.raises(GgufRunError, match="already started"):
            runner.start()
    finally:
        runner.stop()


def test_a_server_that_will_not_start_is_loud(runner, monkeypatch):
    def boom(*_a, **_k):
        raise OSError("permission denied")

    monkeypatch.setattr(subprocess, "Popen", boom)
    with pytest.raises(GgufRunError, match="would not start: permission denied"):
        runner.start()


def test_a_server_that_exits_before_it_is_ready_reports_its_status_and_its_log(runner, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _Proc(exits=1))
    _serve(monkeypatch, {}, health=urllib.error.URLError("refused"))
    said = []
    runner.log = said.append
    with pytest.raises(GgufRunError) as e:
        runner.start()
    assert "exited with status 1" in str(e.value)
    assert "nothing was measured" in str(e.value)
    # The handle has to be closed on the way out. It was not, and the suite's ResourceWarning
    # gate found it as an unclosed file on an unrelated test, which is how that gate always
    # reports: on whichever test the collector happened to reach.
    assert runner._log_file.closed, "the server's log handle was left open on the way out"
    assert any("log is kept at" in line for line in said)


def test_a_server_that_never_answers_health_gives_up_on_a_deadline(runner, monkeypatch):
    """No blocking wait without a deadline, which is baseline section 2.1 with no exception."""
    runner.health_deadline = 0.5
    _serve(monkeypatch, {}, health=urllib.error.URLError("connection refused"))
    with pytest.raises(GgufRunError) as e:
        runner.start()
    said = str(e.value)
    assert "did not answer /health" in said
    assert "abandoned rather than left waiting" in said
    assert runner._proc is None, "the deadline did not stop the process it gave up on"


def test_a_health_answer_that_is_not_200_keeps_waiting_then_gives_up(runner, monkeypatch):
    runner.health_deadline = 0.5
    _serve(monkeypatch, {}, health=503)
    with pytest.raises(GgufRunError, match="did not answer /health"):
        runner.start()


def test_a_server_that_ignores_terminate_is_killed(runner, monkeypatch):
    proc = _Proc(ignores_terminate=True)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
    _serve(monkeypatch, {})
    said = []
    runner.log = said.append
    runner.start()
    runner.stop()
    assert proc.terminated and proc.killed
    assert any("ignored terminate" in line for line in said)


def test_stopping_twice_is_harmless(runner, monkeypatch):
    _serve(monkeypatch, {})
    runner.start()
    runner.stop()
    runner.stop()


def test_a_clean_run_leaves_no_log_file_behind(runner, monkeypatch):
    import os

    _serve(monkeypatch, {"/completion": {"content": "x"}})
    runner.start()
    path = runner._log_path
    assert os.path.exists(path)
    runner.stop()
    assert not os.path.exists(path), "a temporary file nobody deletes is a .part left behind"


def test_a_failed_run_keeps_its_log_and_says_where(runner, monkeypatch):
    import os

    runner.health_deadline = 0.3
    said = []
    runner.log = said.append
    _serve(monkeypatch, {}, health=urllib.error.URLError("refused"))
    with pytest.raises(GgufRunError):
        runner.start()
    kept = [line for line in said if "log is kept at" in line]
    assert kept, "a failure threw away the only diagnostic it had"
    path = kept[0].split("log is kept at ")[1].strip()
    assert os.path.exists(path)
    os.unlink(path)


# ── talking to a started server ──────────────────────────────────────────────────
def test_asking_before_starting_is_refused(runner):
    with pytest.raises(GgufRunError, match="not started"):
        runner.reply("rendered")


def test_an_http_error_names_the_code_and_what_the_server_said(runner, monkeypatch):
    import io

    err = urllib.error.HTTPError("http://x/completion", 400, "Bad Request", {},
                                 io.BytesIO(b'{"error":"context too long"}'))
    _serve(monkeypatch, {"/completion": err})
    try:
        with runner as r, pytest.raises(GgufRunError) as e:
            r.reply("rendered")
        assert "HTTP 400" in str(e.value)
        assert "context too long" in str(e.value)
    finally:
        # The suite runs with -W error::ResourceWarning, and an HTTPError holds an open file
        # object. Leaving it to the collector would put a warning on whichever test ran next.
        err.close()


def test_a_server_that_went_away_mid_run_says_that_it_is_gone(runner, monkeypatch):
    proc = _Proc()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
    _serve(monkeypatch, {"/completion": urllib.error.URLError("broken pipe")})
    runner.start()
    proc.returncode = -9
    with pytest.raises(GgufRunError) as e:
        runner.reply("rendered")
    assert "is gone" in str(e.value)
    runner.stop()


def test_a_server_still_running_but_not_answering_says_that_too(runner, monkeypatch):
    _serve(monkeypatch, {"/completion": TimeoutError("timed out")})
    with runner as r, pytest.raises(GgufRunError) as e:
        r.reply("rendered")
    assert "still running" in str(e.value)


def test_an_answer_that_is_not_json_is_refused(runner, monkeypatch):
    _serve(monkeypatch, {"/completion": b"<html>gateway</html>"})
    with runner as r, pytest.raises(GgufRunError, match="not JSON"):
        r.reply("rendered")


def test_a_missing_reply_is_never_scored_as_an_empty_one(runner, monkeypatch):
    """An empty reply is compliance to every detector in metrics, so this must not degrade."""
    _serve(monkeypatch, {"/completion": {"timings": {}}})
    with runner as r, pytest.raises(GgufRunError, match="count as compliance"):
        r.reply("rendered")


@pytest.mark.parametrize("payload", [{"prompt": ""}, {"prompt": None}, {}])
def test_a_model_with_no_chat_template_cannot_be_scored_comparably(runner, monkeypatch, payload):
    _serve(monkeypatch, {"/apply-template": payload})
    with runner as r, pytest.raises(GgufRunError, match="no way to know what it would have read"):
        r.rendered("hi")


def test_no_prompts_is_refused_rather_than_scored_as_zero(runner, monkeypatch):
    _serve(monkeypatch, {})
    with runner as r, pytest.raises(GgufRunError, match="nothing to score"):
        r.replies([])


def test_a_long_run_emits_a_heartbeat(runner, monkeypatch):
    _serve(monkeypatch, {"/apply-template": {"prompt": "p"}, "/completion": {"content": "c"}})
    said = []
    runner.log = said.append
    with runner as r:
        r.replies(["a", "b", "c"], heartbeat=0.0)
    assert [line for line in said if "prompts" in line], "a long loop with no heartbeat"


def test_rendering_can_be_skipped_for_a_prompt_already_rendered(runner, monkeypatch):
    seen = _serve(monkeypatch, {"/completion": {"content": "c"}})
    with runner as r:
        r.replies(["already rendered"], render=False)
    assert "/apply-template" not in seen


# ── the agreement control ────────────────────────────────────────────────────────
REFUSE = "I'm sorry, but I can't help with that."
COMPLY = "Step one: gather the following materials and begin."


def _arms(n_refusals, n, *, text=REFUSE):
    return [text] * n_refusals + [COMPLY] * (n - n_refusals)


def test_two_paths_that_agree_are_called_comparable():
    hf = _arms(10, 40)
    finding = ggufrun.agreement(hf, list(hf), lossless=True)
    assert finding["comparable"] is True
    assert finding["why_not"] == []
    assert finding["per_prompt_agreement"] == 1.0
    assert finding["identical_replies"] == 1.0
    assert finding["difference"] == 0.0
    assert finding["paired_difference"]["crosses_zero"] is True


def test_a_run_below_the_reporting_floor_cannot_say_whether_they_agree():
    hf = _arms(2, 10)
    finding = ggufrun.agreement(hf, list(hf), lossless=True)
    assert finding["comparable"] is False
    assert any(str(metrics.MIN_REPORTABLE_N) in why for why in finding["why_not"])


def test_rates_that_differ_beyond_the_paired_interval_are_refused():
    hf = _arms(0, 40)
    gg = _arms(40, 40)
    finding = ggufrun.agreement(hf, gg, lossless=True)
    assert finding["comparable"] is False
    assert any("does not contain zero" in why for why in finding["why_not"])
    assert finding["paired_difference"]["crosses_zero"] is False


def test_a_lossless_file_whose_verdicts_scatter_is_refused_even_when_the_rates_match():
    """Equal rates with disagreement underneath is a real and separate failure.

    Twenty prompts refused by one path and twenty different ones refused by the other give
    identical rates and a difference of exactly zero, and the two paths are not running the same
    model. The rate alone cannot see it; the per-prompt check is what does.
    """
    hf = _arms(20, 40)
    gg = [COMPLY] * 20 + [REFUSE] * 20
    finding = ggufrun.agreement(hf, gg, lossless=True)
    assert finding["difference"] == 0.0
    assert finding["paired_difference"]["crosses_zero"] is True
    assert finding["comparable"] is False
    assert any("not running the same model" in why for why in finding["why_not"])
    assert finding["per_prompt_agreement"] == 0.0


def test_a_quantised_file_is_not_held_to_the_per_prompt_floor():
    """Quantisation genuinely changes replies, so a floor there would be invented."""
    hf = _arms(20, 40)
    gg = [COMPLY] * 20 + [REFUSE] * 20
    finding = ggufrun.agreement(hf, gg, lossless=False)
    assert finding["per_prompt_floor_applied"] is False
    assert finding["comparable"] is True, (
        "a quantised file was refused on a bar this project has not measured")


def test_the_disagreeing_prompts_are_named_and_a_few_are_shown():
    prompts = [f"prompt {i}" for i in range(40)]
    hf = _arms(20, 40)
    gg = _arms(18, 40)
    finding = ggufrun.agreement(hf, gg, lossless=True, prompts=prompts)
    assert finding["disagreeing_prompts"] == [18, 19]
    assert len(finding["examples"]) == 2
    assert finding["examples"][0]["prompt"] == "prompt 18"
    assert finding["examples"][0]["transformers"] == REFUSE


def test_examples_are_left_out_when_the_prompts_do_not_line_up():
    finding = ggufrun.agreement(_arms(20, 40), _arms(18, 40), lossless=True, prompts=["one"])
    assert "examples" not in finding


def test_two_paths_of_different_lengths_are_not_an_agreement_figure():
    with pytest.raises(GgufRunError, match="one reply per prompt"):
        ggufrun.agreement(["a", "b"], ["a"], lossless=True)


def test_one_prompt_has_nothing_to_resample():
    finding = ggufrun.agreement([REFUSE], [REFUSE], lossless=True)
    assert finding["paired_difference"] is None
    assert finding["comparable"] is False


def test_nothing_at_all_is_not_a_measurement():
    finding = ggufrun.agreement([], [], lossless=True)
    assert finding["n"] == 0
    assert finding["identical_replies"] is None
    assert finding["per_prompt_agreement"] is None
    assert finding["comparable"] is False


def test_identical_rates_can_still_hide_different_text():
    """The diagnostic that works when the refusal rate is zero on both sides."""
    hf = [COMPLY] * 40
    gg = [COMPLY] * 39 + ["Step one: gather these materials and start."]
    finding = ggufrun.agreement(hf, gg, lossless=True)
    assert finding["per_prompt_agreement"] == 1.0
    assert finding["identical_replies"] == 0.975
    assert finding["comparable"] is True


# ── refusing to quote the two side by side ───────────────────────────────────────
def test_a_comparable_finding_passes_the_gate():
    ggufrun.refuse_if_incomparable({"comparable": True})


def test_an_incomparable_finding_refuses_and_names_every_reason():
    with pytest.raises(GgufRunError) as e:
        ggufrun.refuse_if_incomparable({"comparable": False, "why_not": ["first", "second"]})
    said = str(e.value)
    assert "first" in said and "second" in said
    assert "must not be quoted beside one" in said
    assert "still stands as a measurement of the GGUF on its own terms" in said, (
        "the refusal reads as though the GGUF figure were worthless, which it is not")


def test_a_finding_with_no_recorded_reason_still_refuses():
    with pytest.raises(GgufRunError, match="no reason recorded"):
        ggufrun.refuse_if_incomparable({"comparable": False})


# ── the conditions two runs have to share ────────────────────────────────────────
def test_two_runs_under_the_same_conditions_have_nothing_to_report():
    assert ggufrun.comparability({"prompt": "p", "max_new": 64},
                                 {"prompt": "p", "max_new": 64}) == []


def test_a_differing_condition_is_named_rather_than_averaged_away():
    bad = ggufrun.comparability({"prompt": "a", "max_new": 64}, {"prompt": "b", "max_new": 32})
    assert [field for field, *_ in bad] == ["prompt", "max_new"]
    assert all(why for *_, why in bad), "a named disagreement with no reason beside it"


# ── how it reads ─────────────────────────────────────────────────────────────────
def test_the_description_states_the_verdict_and_the_numbers_behind_it():
    hf = _arms(10, 40)
    lines = ggufrun.describe(ggufrun.agreement(hf, list(hf), lossless=True))
    text = "\n".join(lines)
    assert "lossless file" in text
    assert "contains zero" in text
    assert "COMPARABLE" in text
    assert "identical replies" in text


def test_the_description_of_a_refusal_carries_the_reasons():
    lines = ggufrun.describe(ggufrun.agreement(_arms(0, 40), _arms(40, 40), lossless=True))
    text = "\n".join(lines)
    assert "NOT COMPARABLE" in text
    assert "the finding rather than a failure" in text
    assert "excludes zero" in text


def test_a_description_without_an_interval_does_not_invent_one():
    lines = ggufrun.describe(ggufrun.agreement([REFUSE], [REFUSE], lossless=True))
    assert not any("interval" in line for line in lines)


# ── the last three error paths, which are all "tidying up went wrong" ────────────
def test_a_path_that_is_a_directory_carries_the_reason_it_could_not_be_read(tmp_path):
    # THE OS IS ASKED WHAT IT SAYS rather than told. POSIX gives EISDIR, "Is a directory";
    # Windows refuses the same open with "Access is denied". Hardcoding the POSIX string tested
    # the platform instead of the refusal, and failed on Windows for five days.
    target = tmp_path / "a-directory.gguf"
    target.mkdir()
    try:
        with open(target, "rb"):
            pytest.skip("this platform opens a directory, so there is no reason to pass on")
    except OSError as refused:
        reason = refused.strerror
    with pytest.raises(GgufRunError) as e:
        ggufrun.model_identity(target)
    assert reason in str(e.value), (
        "the refusal does not pass on the errno, so the reader has to guess")


def test_a_file_that_vanishes_between_the_header_and_the_hash_is_not_measured(tmp_path, monkeypatch):
    """A hash read after a header read is a sequence, so the gap between them is real."""
    gguf = _gguf(tmp_path, ftype=1)

    def gone(*_a, **_k):
        raise OSError("No such file or directory")

    monkeypatch.setattr(ggufrun, "_sha256", gone)
    with pytest.raises(GgufRunError, match="could not be read to the end"):
        ggufrun.model_identity(gguf)


def test_a_log_that_cannot_be_removed_is_a_note_and_not_a_failure(runner, monkeypatch):
    """A finished measurement is not turned into an error by a temporary file that will not go."""
    import os

    _serve(monkeypatch, {})
    said = []
    runner.log = said.append
    runner.start()
    monkeypatch.setattr(os, "unlink", lambda *_a: (_ for _ in ()).throw(OSError("busy")))
    runner.stop()
    assert any("could not be removed" in line for line in said)


def test_a_log_that_cannot_be_read_still_produces_the_message_it_was_needed_for(runner, monkeypatch):
    runner.health_deadline = 0.3
    _serve(monkeypatch, {}, health=urllib.error.URLError("refused"))
    runner.log = lambda *_: None
    real_open = open

    def unreadable(path, *a, **k):
        if "senbonzakura-llama-server-" in str(path):
            raise OSError("permission denied")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", unreadable)
    with pytest.raises(GgufRunError) as e:
        runner.start()
    assert "its log could not be read" in str(e.value)
    assert "did not answer /health" in str(e.value)


# ── the compass over a GGUF: reading the verdict position ────────────────────────
def _logprob_answer(pairs):
    """A `/completion` answer carrying these (id, logprob) pairs as its top tokens."""
    return {"completion_probabilities": [{
        "top_logprobs": [{"id": i, "token": f"t{i}", "logprob": lp} for i, lp in pairs]}]}


def _sequence(monkeypatch, answers, *, tokenise=None):
    """Answer `/completion` from a list, one per call, and `/tokenize` from a map."""
    calls = {"completion": 0, "payloads": []}

    def urlopen(req, timeout=None):
        if isinstance(req, str):
            return _Response({"status": "ok"})
        path = "/" + req.full_url.split("/", 3)[3]
        payload = json.loads(req.data)
        if path == "/tokenize":
            return _Response({"tokens": (tokenise or {}).get(payload["content"], [])})
        calls["payloads"].append(payload)
        answer = answers[min(calls["completion"], len(answers) - 1)]
        calls["completion"] += 1
        return _Response(answer)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return calls


def test_the_verdict_tokens_are_read_from_the_guufs_own_tokeniser(runner, monkeypatch):
    """A transformers id against a GGUF vocabulary reads a different token entirely."""
    _sequence(monkeypatch, [_logprob_answer([])],
              tokenise={"HARMFUL": [56, 99], " HARMFUL": [407], "Harmful": [56],
                        " Harmful": [], "harmful": [5794], " harmful": [5794]})
    with runner as r:
        assert ggufrun.verdict_token_ids(r, "HARMFUL") == [56, 407, 5794]


def test_a_word_the_tokeniser_cannot_spell_is_refused(runner, monkeypatch):
    _sequence(monkeypatch, [_logprob_answer([])], tokenise={})
    with runner as r, pytest.raises(ggufrun.GgufCompassError, match="no verdict token"):
        ggufrun.verdict_token_ids(r, "HARMFUL")


def test_a_cheap_request_that_holds_both_verdict_sets_is_not_escalated(runner, monkeypatch):
    calls = _sequence(monkeypatch, [_logprob_answer([(1, -0.5), (2, -2.0), (3, -9.0)])])
    with runner as r:
        readout = ggufrun.verdict_logprobs(r, "<|user|>p", [1], [2])
    assert readout.escalated is False
    assert readout.margin == pytest.approx(1.5)
    assert calls["completion"] == 1, "the whole vocabulary was asked for and did not need to be"
    assert calls["payloads"][0]["n_probs"] == ggufrun.FIRST_TRY_PROBS
    assert calls["payloads"][0]["samplers"] == ["temperature"]


def test_a_missing_verdict_set_escalates_to_the_whole_vocabulary(runner, monkeypatch):
    calls = _sequence(monkeypatch, [_logprob_answer([(1, -0.5)]),
                                    _logprob_answer([(1, -0.5), (2, -3.5)])])
    with runner as r:
        readout = ggufrun.verdict_logprobs(r, "<|user|>p", [1], [2])
    assert readout.escalated is True
    assert readout.margin == pytest.approx(3.0)
    assert calls["completion"] == 2
    assert calls["payloads"][1]["n_probs"] > ggufrun.FIRST_TRY_PROBS


def test_a_verdict_absent_even_from_the_whole_vocabulary_has_no_margin(runner, monkeypatch):
    """None is the honest answer and must not be a zero: zero is a real margin."""
    _sequence(monkeypatch, [_logprob_answer([(1, -0.5)])])
    with runner as r:
        readout = ggufrun.verdict_logprobs(r, "<|user|>p", [1], [2])
    assert readout.margin is None
    assert readout.escalated is True


def test_a_position_with_no_token_probabilities_at_all_is_refused(runner, monkeypatch):
    _sequence(monkeypatch, [{"completion_probabilities": [{"top_logprobs": []}]}])
    with runner as r, pytest.raises(ggufrun.GgufCompassError, match="no verdict position"):
        ggufrun.verdict_logprobs(r, "<|user|>p", [1], [2])


def test_the_verdict_mass_is_the_probability_the_two_sets_hold():
    readout = ggufrun.VerdictReadout({1: -1.0}, {2: -2.0}, top="t", returned=5, escalated=False)
    import math
    assert readout.verdict_mass == pytest.approx(math.exp(-1.0) + math.exp(-2.0))
    assert readout.margin == pytest.approx(1.0)


def test_the_margin_takes_the_best_spelling_in_each_set():
    readout = ggufrun.VerdictReadout({1: -3.0, 2: -1.0}, {3: -5.0, 4: -4.0},
                                     top="t", returned=9, escalated=False)
    assert readout.margin == pytest.approx(3.0)


def test_a_pass_over_several_prompts_reports_how_it_read_them(runner, monkeypatch):
    _sequence(monkeypatch, [_logprob_answer([(1, -0.5), (2, -1.5)])])
    said = []
    with runner as r:
        rows, how = ggufrun.compass_margins(r, ["a", "b", "c"], [1], [2], heartbeat=0.0,
                                            log=said.append)
    assert [row["margin"] for row in rows] == [pytest.approx(1.0)] * 3
    assert how == {"escalated": 0, "n": 3, "first_try_probs": ggufrun.FIRST_TRY_PROBS}
    assert said, "a long loop with no heartbeat"


# ── the compass agreement control ────────────────────────────────────────────────
def _separated(n, *, offset=0.0):
    harmful = [1.0 + offset + i * 0.01 for i in range(n)]
    benign = [-1.0 + offset - i * 0.01 for i in range(n)]
    return harmful, benign


def test_two_compasses_that_order_the_prompts_alike_are_comparable():
    h, b = _separated(20)
    finding = ggufrun.compass_agreement(h, b, [x * 3 for x in h], [x * 3 for x in b])
    assert finding["transformers_auc"] == 1.0
    assert finding["gguf_auc"] == 1.0
    assert finding["auc_difference"] == 0.0
    assert finding["comparable"] is True


def test_the_comparison_is_on_the_auc_and_so_survives_a_scale_difference():
    """A quantised file's logits are not the original's, and the AUC does not care."""
    h, b = _separated(20)
    finding = ggufrun.compass_agreement(h, b, [x * 100 + 7 for x in h], [x * 100 + 7 for x in b])
    assert finding["comparable"] is True
    assert finding["auc_difference"] == 0.0


def test_two_compasses_that_order_the_prompts_differently_are_refused():
    h, b = _separated(20)
    finding = ggufrun.compass_agreement(h, b, b, h)
    assert finding["gguf_auc"] == 0.0
    assert finding["comparable"] is False
    assert any("order these prompts differently" in why for why in finding["why_not"])


def test_a_compass_control_below_the_reporting_floor_says_so():
    h, b = _separated(5)
    finding = ggufrun.compass_agreement(h, b, h, b)
    assert finding["comparable"] is False
    assert any(str(metrics.MIN_REPORTABLE_N) in why for why in finding["why_not"])


def test_an_unreadable_prompt_in_either_arm_refuses_the_whole_figure():
    h, b = _separated(20)
    with pytest.raises(ggufrun.GgufCompassError, match="no readable verdict"):
        ggufrun.compass_agreement([*h[:-1], None], b, h, b)


def test_two_arms_of_different_lengths_are_not_a_paired_comparison():
    h, b = _separated(20)
    with pytest.raises(ggufrun.GgufCompassError, match="same prompts in the same order"):
        ggufrun.compass_agreement(h, b, h[:-1], b)


def test_a_gguf_is_recognised_by_its_first_four_bytes(tmp_path):
    """Not by its name: `--model` already takes three kinds of input, and a flag would be a
    fourth way to say the same thing.
    """
    good = tmp_path / "not-named-like-one.bin"
    good.write_bytes(b"GGUF\x03\x00\x00\x00")
    bad = tmp_path / "pretending.gguf"
    bad.write_bytes(b"PK\x03\x04 a zip file")
    assert ggufrun.looks_like_gguf(good) is True
    assert ggufrun.looks_like_gguf(bad) is False
    assert ggufrun.looks_like_gguf(tmp_path / "absent.gguf") is False
    assert ggufrun.looks_like_gguf(tmp_path) is False


def test_a_short_pass_says_nothing_until_the_heartbeat_is_due(runner, monkeypatch):
    _sequence(monkeypatch, [_logprob_answer([(1, -0.5), (2, -1.5)])])
    said = []
    with runner as r:
        ggufrun.compass_margins(r, ["a"], [1], [2], heartbeat=1e6, log=said.append)
    assert said == [], "a two-prompt pass printed progress nobody needed"
