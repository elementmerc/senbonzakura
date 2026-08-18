"""`senbonzakura imatrix`, and why the calibration set is part of the result.

An importance matrix decides which weights keep their precision when a model is quantised, and it
decides it FROM text. So "quantised with an imatrix" is not one thing; it is one thing per
calibration set, and a comparison that mixes an `i1-` file with a plain one is measuring the
quantiser as well as the model.

That is not hypothetical. A sister project's 8B comparison put an `i1-Q4_K_M` abliterated arm
against a plain `Q4_K_M` stock arm and read the difference as an effect of the edit. The only place
that difference was written down was the filename, which is why every matrix built here carries a
sidecar saying what it was calibrated on.
"""
import json
import pathlib

import pytest

from senbonzakura import imatrix
from senbonzakura.imatrix import ImatrixError


# ── the calibration set, which is the part that carries meaning ──────────────────
def test_a_bundled_corpus_is_resolved_and_its_provenance_recorded():
    text, prov = imatrix.calibration_text(corpus="advbench")
    assert text.strip()
    assert prov["source"] == "bundled corpus"
    assert prov["corpus"] == "advbench"
    assert prov["arm"] == "harmful"
    # The upstream pin travels with it: "calibrated on advbench" is not enough to reproduce.
    assert prov["upstream"] and prov["commit"]


def test_the_harmful_arm_is_recorded_so_it_can_be_warned_about():
    """A matrix calibrated on harmful prompts preserves the machinery those prompts exercise,
    which on a refusal-abliteration tool is not a neutral act. The tool cannot decide that for
    the user, so it must at least be able to say it.
    """
    _, prov = imatrix.calibration_text(corpus="advbench")
    assert prov["arm"] == "harmful"
    _, benign = imatrix.calibration_text(corpus="xstest-safe")
    assert benign["arm"] == "benign"


def test_a_file_is_accepted_and_its_path_recorded(tmp_path):
    f = tmp_path / "cal.txt"
    f.write_text("some broad prose to calibrate on", encoding="utf-8")
    text, prov = imatrix.calibration_text(path=f)
    assert "broad prose" in text
    assert prov["source"] == "file" and prov["bytes"] > 0


def test_an_empty_calibration_file_is_refused(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("   \n", encoding="utf-8")
    with pytest.raises(ImatrixError, match="nothing to calibrate on"):
        imatrix.calibration_text(path=f)


def test_a_missing_calibration_file_is_refused(tmp_path):
    with pytest.raises(ImatrixError, match="no calibration file"):
        imatrix.calibration_text(path=tmp_path / "nope.txt")


def test_an_unknown_corpus_says_what_there_is():
    with pytest.raises(ImatrixError, match="advbench"):
        imatrix.calibration_text(corpus="advbenchh")


def test_the_ambiguous_corpus_name_is_still_refused_here():
    """`xstest` merges prompts a model should answer with ones it should not. Calibrating on that
    mixture is as meaningless as scoring refusal on it.
    """
    with pytest.raises(ImatrixError):
        imatrix.calibration_text(corpus="xstest")


# ── the pre-flight ───────────────────────────────────────────────────────────────
def _gguf(path, ftype="BF16"):
    path.write_bytes(b"GGUF" + b"\0" * 32)
    return path


def test_an_already_quantised_model_is_refused(tmp_path, monkeypatch):
    """An importance matrix guides how to quantise full-precision weights. Computing one from
    already-quantised weights measures the damage instead of guiding it.
    """
    src = _gguf(tmp_path / "m.gguf")
    monkeypatch.setattr(imatrix.gguf_io, "verify",
                        lambda *a, **k: {"file_type": "Q4_K_M", "architecture": "qwen3"})
    with pytest.raises(ImatrixError, match="already a Q4_K_M"):
        imatrix.preflight(src, tmp_path / "o.gguf", force=False)


def test_a_full_precision_model_passes(tmp_path, monkeypatch):
    src = _gguf(tmp_path / "m.gguf")
    monkeypatch.setattr(imatrix.gguf_io, "verify",
                        lambda *a, **k: {"file_type": "BF16", "architecture": "lfm2"})
    got = imatrix.preflight(src, tmp_path / "o.gguf", force=False)
    assert got["architecture"] == "lfm2"


def test_writing_over_the_model_is_refused(tmp_path, monkeypatch):
    src = _gguf(tmp_path / "m.gguf")
    monkeypatch.setattr(imatrix.gguf_io, "verify",
                        lambda *a, **k: {"file_type": "BF16", "architecture": "x"})
    with pytest.raises(ImatrixError, match="output path is the model"):
        imatrix.preflight(src, src, force=False)


def test_an_existing_output_needs_force(tmp_path, monkeypatch):
    src = _gguf(tmp_path / "m.gguf")
    out = _gguf(tmp_path / "o.gguf")
    monkeypatch.setattr(imatrix.gguf_io, "verify",
                        lambda *a, **k: {"file_type": "BF16", "architecture": "x"})
    with pytest.raises(ImatrixError, match="Pass --force"):
        imatrix.preflight(src, out, force=False)
    imatrix.preflight(src, out, force=True)


def test_a_missing_model_is_refused(tmp_path):
    with pytest.raises(ImatrixError, match="no model at"):
        imatrix.preflight(tmp_path / "nope.gguf", tmp_path / "o.gguf", force=False)


def test_an_unreadable_gguf_is_a_plain_failure(tmp_path, monkeypatch):
    src = _gguf(tmp_path / "m.gguf")
    monkeypatch.setattr(imatrix.gguf_io, "verify",
                        lambda *a, **k: (_ for _ in ()).throw(imatrix.gguf_io.GGUFError("bad magic")))
    with pytest.raises(ImatrixError, match="cannot read"):
        imatrix.preflight(src, tmp_path / "o.gguf", force=False)


# ── naming and the sidecar ───────────────────────────────────────────────────────
def test_the_default_output_is_named_after_the_model(tmp_path):
    assert imatrix.default_output("lfm2-brain-bf16.gguf").name == "lfm2-brain-bf16-imatrix.gguf"


def test_describe_reads_a_sidecar(tmp_path):
    m = tmp_path / "im.gguf"
    m.write_bytes(b"x")
    (tmp_path / ("im.gguf" + imatrix.SIDECAR_SUFFIX)).write_text(
        json.dumps({"calibration": {"corpus": "advbench"}}), encoding="utf-8")
    got = imatrix.describe(m)
    assert got["calibration"]["corpus"] == "advbench"


def test_describe_returns_none_when_provenance_is_absent(tmp_path):
    """A matrix from elsewhere has no sidecar, and the honest answer is "unknown" rather than a
    guess assembled from its filename.
    """
    m = tmp_path / "someone-elses.gguf"
    m.write_bytes(b"x")
    assert imatrix.describe(m) is None


def test_describe_survives_a_corrupt_sidecar(tmp_path):
    m = tmp_path / "im.gguf"
    m.write_bytes(b"x")
    (tmp_path / ("im.gguf" + imatrix.SIDECAR_SUFFIX)).write_text("{not json", encoding="utf-8")
    assert imatrix.describe(m) is None


# ── run() ────────────────────────────────────────────────────────────────────────
class _Ran:
    def __init__(self, rc=0, write=None):
        self.rc, self.write, self.calls = rc, write, []

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        if self.write is not None:
            out = pathlib.Path(argv[argv.index("-o") + 1])
            out.write_bytes(self.write)
        return type("R", (), {"returncode": self.rc})()


def _ok(monkeypatch, ran):
    monkeypatch.setattr(imatrix.gguf_io, "verify",
                        lambda *a, **k: {"file_type": "BF16", "architecture": "lfm2"})
    monkeypatch.setattr(imatrix, "find_binary", lambda _n, **k: ("/x/llama-imatrix", "vendored"))
    monkeypatch.setattr(imatrix.subprocess, "run", ran)


def test_a_successful_run_writes_the_sidecar_beside_the_matrix(tmp_path, monkeypatch):
    src = _gguf(tmp_path / "m.gguf")
    out = tmp_path / "im.gguf"
    ran = _Ran(rc=0, write=b"matrix")
    _ok(monkeypatch, ran)
    assert imatrix.run([str(src), "-o", str(out), "--corpus", "advbench"], log=lambda _m: None) == 0
    side = tmp_path / ("im.gguf" + imatrix.SIDECAR_SUFFIX)
    assert side.is_file()
    d = json.loads(side.read_text())
    assert d["calibration"]["corpus"] == "advbench"
    assert d["architecture"] == "lfm2"


def test_the_harmful_corpus_warning_is_printed_every_time(tmp_path, monkeypatch):
    src = _gguf(tmp_path / "m.gguf")
    ran = _Ran(rc=0, write=b"matrix")
    _ok(monkeypatch, ran)
    msgs = []
    imatrix.run([str(src), "-o", str(tmp_path / "im.gguf"), "--corpus", "advbench"],
                log=msgs.append)
    assert any("not a neutral choice" in m for m in msgs), msgs


def test_a_benign_corpus_does_not_print_the_harmful_warning(tmp_path, monkeypatch):
    src = _gguf(tmp_path / "m.gguf")
    ran = _Ran(rc=0, write=b"matrix")
    _ok(monkeypatch, ran)
    msgs = []
    imatrix.run([str(src), "-o", str(tmp_path / "im.gguf"), "--corpus", "xstest-safe"],
                log=msgs.append)
    assert not any("not a neutral choice" in m for m in msgs)


def test_a_failed_run_leaves_no_matrix_behind(tmp_path, monkeypatch):
    src = _gguf(tmp_path / "m.gguf")
    out = tmp_path / "im.gguf"
    _ok(monkeypatch, _Ran(rc=1, write=b"partial"))
    with pytest.raises(SystemExit, match="exited 1"):
        imatrix.run([str(src), "-o", str(out), "--corpus", "advbench"], log=lambda _m: None)
    assert not out.exists()


def test_the_calibration_temp_file_is_cleaned_up(tmp_path, monkeypatch):
    """It writes the corpus to a temp file for the binary. Leaving harmful prompts in /tmp is not
    something to do by accident.
    """
    src = _gguf(tmp_path / "m.gguf")
    ran = _Ran(rc=0, write=b"matrix")
    _ok(monkeypatch, ran)
    imatrix.run([str(src), "-o", str(tmp_path / "im.gguf"), "--corpus", "advbench"],
                log=lambda _m: None)
    from pathlib import Path
    cal = ran.calls[0][ran.calls[0].index("-f") + 1]
    assert not Path(cal).exists(), "the calibration text was left on disk"


def test_the_chunk_count_and_gpu_layers_reach_the_binary(tmp_path, monkeypatch):
    src = _gguf(tmp_path / "m.gguf")
    ran = _Ran(rc=0, write=b"matrix")
    _ok(monkeypatch, ran)
    imatrix.run([str(src), "-o", str(tmp_path / "im.gguf"), "--corpus", "advbench",
                 "--chunks", "17", "--gpu-layers", "5"], log=lambda _m: None)
    argv = ran.calls[0]
    assert argv[argv.index("--chunks") + 1] == "17"
    assert argv[argv.index("-ngl") + 1] == "5"


def test_a_corpus_and_a_file_cannot_both_be_given(tmp_path):
    with pytest.raises(SystemExit):
        imatrix.build_parser().parse_args(["m.gguf", "--corpus", "advbench", "--file", "x.txt"])


def test_one_of_them_must_be_given():
    """There is no default calibration set on purpose: the choice changes the result."""
    with pytest.raises(SystemExit):
        imatrix.build_parser().parse_args(["m.gguf"])


def test_no_disk_space_is_caught_before_the_job(tmp_path, monkeypatch):
    src = _gguf(tmp_path / "m.gguf")
    monkeypatch.setattr(imatrix.gguf_io, "verify",
                        lambda *a, **k: {"file_type": "BF16", "architecture": "x"})
    monkeypatch.setattr(imatrix, "free_bytes_for", lambda _p: 1)
    with pytest.raises(ImatrixError, match="free"):
        imatrix.preflight(src, tmp_path / "o.gguf", force=False)


def test_a_missing_binary_is_a_plain_failure(tmp_path, monkeypatch):
    src = _gguf(tmp_path / "m.gguf")
    monkeypatch.setattr(imatrix.gguf_io, "verify",
                        lambda *a, **k: {"file_type": "BF16", "architecture": "x"})
    monkeypatch.setattr(imatrix, "find_binary",
                        lambda _n, **k: (_ for _ in ()).throw(imatrix.VendorError("not vendored")))
    with pytest.raises(SystemExit, match="not vendored"):
        imatrix.run([str(src), "-o", str(tmp_path / "im.gguf"), "--corpus", "advbench"],
                    log=lambda _m: None)


def test_a_preflight_failure_is_translated_not_propagated(tmp_path, monkeypatch):
    """A traceback in front of a readable sentence is not the plain-language failure a user is
    owed, and this is the single boundary where that translation happens.
    """
    with pytest.raises(SystemExit, match="cannot compute an importance matrix"):
        imatrix.run([str(tmp_path / "nope.gguf"), "--corpus", "advbench"], log=lambda _m: None)


def test_main_delegates_to_run(monkeypatch):
    monkeypatch.setattr(imatrix, "run", lambda argv=None: 7)
    assert imatrix.main([]) == 7
