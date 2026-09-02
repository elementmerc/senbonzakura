# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`senbonzakura doctor`: the command that answers "can this install actually do the job".

The test that matters most here is `test_a_broken_module_does_not_leave_an_architecture_marked_
supported`, and it exists because doctor reproduced, in its own output, the exact defect it was
written to catch. The first version printed:

    ✗  architecture modules      22 failed to import: command_r, deepseek, gemma, ...
    ✓  arch Lfm2MoeForCausalLM   supported

three lines apart. Both came from the same source: the converter's static registry, which lists an
architecture whether or not the module providing it imported. A reader skimming for ticks would
have been reassured by the line that was wrong.
"""
import pathlib
import sys

import pytest

from senbonzakura import doctor


def _by_name(checks, prefix):
    return [c for c in checks if c.name.startswith(prefix)]


# ── the statuses mean what the exit code says they mean ──────────────────────────
def test_a_clean_run_exits_zero():
    checks = [doctor._pass("a", "fine"), doctor._pass("b", "fine")]
    assert doctor.report(checks, log=lambda _m: None) == doctor.OK


def test_an_advisory_is_not_a_failure():
    """A CPU-only box is a warning, not a broken install: scoring works, editing is just slow."""
    checks = [doctor._pass("a", "fine"), doctor._warn("torch", "no cuda")]
    assert doctor.report(checks, log=lambda _m: None) == doctor.WARN


def test_a_failure_exits_with_its_own_code():
    checks = [doctor._pass("a", "fine"), doctor._fail("b", "broken", "do the thing")]
    assert doctor.report(checks, log=lambda _m: None) == doctor.FAIL


def test_the_report_prints_the_fix_for_anything_that_is_not_passing():
    """A diagnostic that says what is wrong and not what to do about it is half a diagnostic."""
    lines = []
    doctor.report([doctor._fail("b", "broken", "re-run the vendoring tool")], log=lines.append)
    text = "\n".join(lines)
    assert "broken" in text and "re-run the vendoring tool" in text


def test_a_passing_check_does_not_print_a_fix():
    lines = []
    doctor.report([doctor._pass("a", "fine")], log=lines.append)
    assert "->" not in "\n".join(lines)


# ── the converter check, and the defect doctor itself had ────────────────────────
def test_a_broken_module_does_not_leave_an_architecture_marked_supported(monkeypatch):
    """THE regression. `names` is the converter's static registry; while any module is failing to
    import, membership of that registry means "mentioned", not "works".
    """
    monkeypatch.setattr(doctor, "__name__", doctor.__name__)  # no-op, keeps the patch local
    import senbonzakura.convert as c
    monkeypatch.setattr(c, "supported_architectures",
                        lambda _s, **k: ({"Lfm2MoeForCausalLM", "Qwen3ForCausalLM"}, ["lfm2"]))
    checks = doctor.check_converter()

    modules = _by_name(checks, "architecture modules")
    assert modules and modules[0].status == "fail"

    for c_ in _by_name(checks, "arch "):
        assert c_.status != "pass", (
            f"{c_.name} was marked passing while modules were failing to import, which is the "
            f"static-registry lie this command exists to catch")


def test_every_target_architecture_passes_when_nothing_is_broken(monkeypatch):
    import senbonzakura.convert as c
    monkeypatch.setattr(c, "supported_architectures",
                        lambda _s, **k: ({"Lfm2ForCausalLM", "Lfm2MoeForCausalLM",
                                          "Qwen3ForCausalLM", "LlamaForCausalLM"}, []))
    checks = doctor.check_converter()
    assert all(c_.status == "pass" for c_ in _by_name(checks, "arch "))
    assert _by_name(checks, "architecture modules")[0].status == "pass"


def test_an_architecture_absent_at_this_pin_is_an_advisory_not_a_failure(monkeypatch):
    """A pin that predates an architecture is a scheduling fact, not a broken install."""
    import senbonzakura.convert as c
    monkeypatch.setattr(c, "supported_architectures", lambda _s, **k: ({"Qwen3ForCausalLM"}, []))
    checks = doctor.check_converter()
    lfm2 = [x for x in checks if x.name == "arch Lfm2MoeForCausalLM"]
    assert lfm2 and lfm2[0].status == "warn"


def test_a_converter_reporting_nothing_at_all_is_a_failure(monkeypatch):
    import senbonzakura.convert as c
    monkeypatch.setattr(c, "supported_architectures", lambda _s, **k: (set(), []))
    checks = doctor.check_converter()
    assert checks[0].status == "fail"
    assert "no supported architectures" in checks[0].detail


def test_a_missing_converter_says_which_step_was_not_run(monkeypatch):
    import senbonzakura.vendored as v
    monkeypatch.setattr(doctor, "check_converter", doctor.check_converter)
    monkeypatch.setattr(v, "find_script",
                        lambda _n: (_ for _ in ()).throw(v.VendorError("not fetched; run the tool")))
    checks = doctor.check_converter()
    assert checks[0].status == "fail"


# ── the binary check asks whether it RUNS ────────────────────────────────────────
def test_a_binary_that_exists_and_will_not_start_is_a_failure(monkeypatch):
    """Exactly what the first vendoring attempt produced: present, and missing a shared library.
    A presence check passes it; only running it does not.
    """
    import senbonzakura.vendored as v
    monkeypatch.setattr(v, "find_binary", lambda _n, **k: ("/nope/llama-quantize", "vendored"))
    monkeypatch.setattr(doctor.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no such file")))
    c = doctor.check_quantize()
    assert c.status == "fail"
    assert "will not run" in c.detail


def test_a_binary_that_runs_but_prints_no_usage_is_a_failure(monkeypatch):
    import senbonzakura.vendored as v
    monkeypatch.setattr(v, "find_binary", lambda _n, **k: ("/x/llama-quantize", "vendored"))
    monkeypatch.setattr(doctor.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": b"", "stderr": b"???"})())
    assert doctor.check_quantize().status == "fail"


def test_a_working_binary_passes_and_says_where_it_came_from(monkeypatch):
    import senbonzakura.vendored as v
    monkeypatch.setattr(v, "find_binary", lambda _n, **k: ("/x/llama-quantize", "vendored"))
    monkeypatch.setattr(doctor.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": b"usage: llama-quantize",
                                                       "stderr": b""})())
    c = doctor.check_quantize()
    assert c.status == "pass" and "vendored" in c.detail


# ── the data checks ──────────────────────────────────────────────────────────────
def test_a_corpus_of_the_wrong_size_is_a_failure(monkeypatch):
    from senbonzakura import corpora
    monkeypatch.setattr(doctor, "check_corpora", doctor.check_corpora)
    monkeypatch.setattr(corpora, "load", lambda _k, **kw: ["only one"])
    bad = [c for c in doctor.check_corpora() if c.status == "fail"]
    assert bad, "a pack disagreeing with the registry passed"
    assert "expected" in bad[0].detail


def test_the_real_install_has_all_its_corpora():
    """Not a mock: the shipped pack has to actually decode at its declared sizes."""
    from senbonzakura import corpora
    checks = doctor.check_corpora()
    assert len(checks) == len(corpora.CORPORA)
    assert all(c.status == "pass" for c in checks), [c.detail for c in checks if c.status != "pass"]


# ── the whole thing runs ─────────────────────────────────────────────────────────
def test_run_checks_produces_a_report_and_an_exit_code():
    lines = []
    rc = doctor.report(doctor.run_checks(deep=False), log=lines.append)
    assert rc in (doctor.OK, doctor.WARN, doctor.FAIL)
    text = "\n".join(lines)
    assert "senbonzakura doctor" in text
    assert "checks," in text


def test_the_cli_dispatches_doctor():
    from senbonzakura import cli
    assert "doctor" in cli.DELEGATED
    assert cli._delegate("doctor") is doctor.main


@pytest.mark.parametrize(("status", "mark"), [("pass", "✓"), ("warn", "!"), ("fail", "✗")])
def test_each_status_has_a_distinct_mark(status, mark):
    assert doctor.Check("x", status, "d").mark == mark


# ── the deep check, whose failures matter more than its successes ────────────────
def test_deep_is_skipped_rather_than_failed_when_torch_is_absent(monkeypatch):
    """A machine without torch cannot run the deep check and is not thereby broken."""
    import builtins
    real = builtins.__import__

    def no_torch(name, *a, **k):
        if name == "torch":
            raise ImportError("no torch here")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    out = doctor.deep_check(log=lambda _m: None)
    assert out and out[0].status == "warn"
    assert "skipped" in out[0].detail


def test_deep_skips_cleanly_when_no_tokenizer_can_be_fetched(monkeypatch):
    """The deep check needs one small tokenizer. Offline with a cold cache it must say so rather
    than fail, because that is a fact about the network and not about the install.
    """
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(OSError("offline"))))
    out = doctor.deep_check(log=lambda _m: None)
    assert out and out[0].status == "warn" and "tokenizer" in out[0].detail


def test_a_deep_convert_that_writes_nothing_is_a_failure(monkeypatch):
    from senbonzakura import convert
    monkeypatch.setattr(convert, "run", lambda *a, **k: 0)   # claims success, writes no file
    out = doctor.deep_check(log=lambda _m: None)
    assert any(c.status == "fail" and "convert" in c.name for c in out), [c.name for c in out]


def test_a_deep_convert_that_raises_is_reported_not_propagated(monkeypatch):
    from senbonzakura import convert
    monkeypatch.setattr(convert, "run",
                        lambda *a, **k: (_ for _ in ()).throw(SystemExit("converter died")))
    out = doctor.deep_check(log=lambda _m: None)
    assert any(c.status == "fail" for c in out)


# ── main ─────────────────────────────────────────────────────────────────────────
def test_main_runs_without_deep_by_default(monkeypatch):
    seen = {}
    monkeypatch.setattr(doctor, "run_checks", lambda **k: seen.update(k) or [doctor._pass("x", "y")])
    assert doctor.main([]) == doctor.OK
    assert seen["deep"] is False


def test_main_passes_deep_through(monkeypatch):
    seen = {}
    monkeypatch.setattr(doctor, "run_checks", lambda **k: seen.update(k) or [doctor._pass("x", "y")])
    doctor.main(["--deep"])
    assert seen["deep"] is True


def test_an_unreadable_manifest_is_a_failure_not_a_crash(monkeypatch):
    from senbonzakura import vendoring
    monkeypatch.setattr(vendoring, "load_manifest",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad json")))
    out = doctor.check_pins()
    assert out[0].status == "fail" and "cannot be read" in out[0].detail


def test_an_unrecognised_platform_is_an_advisory(monkeypatch):
    from senbonzakura import vendored
    monkeypatch.setattr(vendored, "platform_key", lambda: None)
    assert doctor.check_platform().status == "warn"


def test_a_missing_bundled_track_is_a_failure(monkeypatch, tmp_path):
    from senbonzakura import bundled
    monkeypatch.setattr(bundled, "data_path", lambda: tmp_path / "nope.bin")
    assert doctor.check_track().status == "fail"


def test_torch_without_cuda_is_an_advisory_and_names_the_version():
    c = doctor.check_torch()
    assert c.status in ("pass", "warn")
    assert any(ch.isdigit() for ch in c.detail), "the torch version is not in the detail line"


def test_the_deep_check_reports_both_steps_when_they_succeed(monkeypatch):
    """The success path of the only check that proves the whole chain. Mocked at the command
    boundary rather than the binary, so it exercises doctor's own reporting rather than llama.cpp.
    """
    from senbonzakura import convert, gguf_io, quantise

    def fake_convert(argv, log=print):
        pathlib.Path(argv[1]).write_bytes(b"GGUF")
        return 0

    def fake_quantise(argv, log=print):
        pathlib.Path(argv[1]).write_bytes(b"GGUF")
        return 0

    monkeypatch.setattr(convert, "run", fake_convert)
    monkeypatch.setattr(quantise, "run", fake_quantise)
    monkeypatch.setattr(gguf_io, "verify",
                        lambda *a, **k: {"tensor_count": 25, "architecture": "qwen3",
                                         "file_type": "BF16"})
    out = doctor.deep_check(log=lambda _m: None)
    names = [c.name for c in out]
    assert "deep convert" in names and "deep quantise" in names
    assert all(c.status == "pass" for c in out), [(c.name, c.detail) for c in out]


def test_a_deep_quantise_that_writes_nothing_is_a_failure(monkeypatch):
    from senbonzakura import convert, gguf_io, quantise
    monkeypatch.setattr(convert, "run",
                        lambda argv, log=print: (pathlib.Path(argv[1]).write_bytes(b"GGUF"), 0)[1])
    monkeypatch.setattr(quantise, "run", lambda argv, log=print: 0)   # claims success, no file
    monkeypatch.setattr(gguf_io, "verify",
                        lambda *a, **k: {"tensor_count": 25, "architecture": "qwen3",
                                         "file_type": "BF16"})
    out = doctor.deep_check(log=lambda _m: None)
    assert any(c.status == "fail" and "quantise" in c.name for c in out)


# ── the fix line has to be a thing the reader can actually do ────────────────────
def test_the_corpus_fix_names_the_github_cli_when_it_is_missing(monkeypatch):
    """`build_corpora.py` fetches through `gh`. Telling someone to run a command that will fail
    on their machine is worse than saying nothing: they follow the instruction, it dies, and the
    instruction was ours. Found by running the script on the CPU box, which has no `gh`.
    """
    monkeypatch.setattr(doctor.shutil if hasattr(doctor, "shutil") else __import__("shutil"),
                        "which", lambda _n: None)
    fix = doctor._corpus_fix()
    assert "build_corpora.py" in fix
    assert "GitHub CLI" in fix and "cli.github.com" in fix


def test_the_corpus_fix_stays_short_when_the_cli_is_there(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda _n: "/usr/bin/gh")
    fix = doctor._corpus_fix()
    assert "build_corpora.py" in fix
    assert "GitHub CLI" not in fix, "the note is only useful when the tool is actually absent"


# ── the page-locked ceiling (from the Soup analysis) ─────────────────────────────────
class _FakeCuda:
    def __init__(self, available=True):
        self._available = available

    def is_available(self):
        return self._available


def _fake_torch(monkeypatch, *, available=True, refuse_after=None, raiser=None):
    """A torch stand-in whose pinned allocation succeeds a set number of times, then refuses."""
    import types
    calls = {"n": 0}

    def empty(_n, dtype=None, pin_memory=False):
        calls["n"] += 1
        if raiser is not None:
            raise raiser
        if refuse_after is not None and calls["n"] > refuse_after:
            raise RuntimeError("CUDA error: cannot allocate pinned memory")
        return bytearray(8)          # a stand-in; only its lifetime matters to the code

    fake = types.SimpleNamespace(empty=empty, uint8="uint8", cuda=_FakeCuda(available),
                                 __version__="fake")
    monkeypatch.setitem(sys.modules, "torch", fake)
    return calls


def test_the_pinned_check_says_why_it_could_not_measure_without_a_card(monkeypatch):
    _fake_torch(monkeypatch, available=False)
    c = doctor.check_pinned_memory()
    assert c.status == "warn"
    assert "no cuda device" in c.detail
    assert "overlap" in c.fix, "it has to say what the number would have been for"


def test_the_pinned_check_reports_a_ceiling_when_the_machine_refuses(monkeypatch):
    _fake_torch(monkeypatch, refuse_after=3)
    c = doctor.check_pinned_memory(max_bytes=doctor.PINNED_STEP_BYTES * 10)
    assert c.status == "pass"
    assert "ceiling" in c.detail
    expected = 3 * doctor.PINNED_STEP_BYTES / 1e9
    assert f"{expected:.1f} GB" in c.detail
    assert "loses copy/compute overlap" in c.detail, "the number needs its consequence beside it"


def test_a_probe_that_runs_out_of_budget_says_at_least(monkeypatch):
    """'At least 8 GB' and 'exactly 8 GB' are different findings and must not be confused."""
    _fake_torch(monkeypatch)
    c = doctor.check_pinned_memory(max_bytes=doctor.PINNED_STEP_BYTES * 4)
    assert c.status == "pass"
    assert "at least" in c.detail
    assert "ceiling" not in c.detail


def test_a_machine_that_cannot_pin_at_all_is_a_warning_with_a_fix(monkeypatch):
    _fake_torch(monkeypatch, refuse_after=0)
    c = doctor.check_pinned_memory()
    assert c.status == "warn"
    assert "cannot pin" in c.detail
    assert "ulimit -l" in c.fix


def test_the_probe_frees_everything_even_when_it_refuses(monkeypatch):
    """Holding gigabytes of page-locked memory past a diagnostic would be worse than the defect."""
    import gc
    _fake_torch(monkeypatch, refuse_after=2)
    before = len(gc.get_objects())
    doctor.check_pinned_memory(max_bytes=doctor.PINNED_STEP_BYTES * 10)
    gc.collect()
    # Not an exact count, which would be flaky; the point is that it does not grow by the number
    # of blocks allocated.
    assert len(gc.get_objects()) < before + 100


def test_an_unexpected_failure_in_the_probe_does_not_take_the_run_down(monkeypatch):
    _fake_torch(monkeypatch, raiser=ValueError("something odd"))
    c = doctor.check_pinned_memory()
    assert c.status == "warn"
    assert "ValueError" in c.detail
    assert "does not affect an ordinary run" in c.fix


def test_the_default_run_probes_cheaply_and_deep_probes_further(monkeypatch):
    """A default `doctor` must not pin gigabytes; finding the real ceiling means climbing to it."""
    seen = {}

    def spy(max_bytes=None):
        seen["max_bytes"] = max_bytes
        return doctor._pass("pinned memory", "stub")

    monkeypatch.setattr(doctor, "check_pinned_memory", spy)
    monkeypatch.setattr(doctor, "check_pins", list)
    monkeypatch.setattr(doctor, "check_quantize", lambda: doctor._pass("q", ""))
    monkeypatch.setattr(doctor, "check_converter", lambda *a, **k: [])
    monkeypatch.setattr(doctor, "check_track", lambda: doctor._pass("t", ""))
    monkeypatch.setattr(doctor, "check_corpora", list)
    doctor.run_checks(deep=False, log=lambda _m: None)
    assert seen["max_bytes"] is None, "the default run must use the cheap one-step probe"
    doctor.run_checks(deep=True, log=lambda _m: None)
    assert seen["max_bytes"] == doctor.PINNED_DEEP_MAX_BYTES
