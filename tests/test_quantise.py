# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`senbonzakura quantise`, including one genuine end-to-end run through the vendored binary.

The end-to-end test builds a small but STRUCTURALLY REAL llama-architecture GGUF and quantises it
to Q4_K_M, because a k-quant is a per-tensor type mix that llama-quantize chooses from the
architecture metadata: a bag of nameless tensors is not a model it will process. It runs in well
under a second and is skipped where no binary has been vendored, which is any fresh checkout.

Everything else here is a guard, and each guard exists because its absence has cost something:
an exit code that was believed over the file it produced, a partial write left looking like a
result, and a lossy step stacked on a lossy step.
"""
from pathlib import Path

import numpy as np
import pytest

from senbonzakura import gguf_io, quantise, vendored
from senbonzakura.vendored import VendorError


def _tiny_gguf(path, *, n_layer=2, n_embd=256, n_ff=512, n_head=4, n_vocab=512, ftype=0):
    """A real llama-arch GGUF. Small, and complete enough for llama-quantize to accept it."""
    from gguf import GGUFWriter
    w = GGUFWriter(str(path), "llama")
    w.add_block_count(n_layer)
    w.add_context_length(128)
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_head)
    w.add_rope_dimension_count(n_embd // n_head)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_file_type(ftype)
    rng = np.random.default_rng(0)

    def t(name, shape):
        w.add_tensor(name, rng.standard_normal(shape, dtype=np.float32))

    t("token_embd.weight", (n_vocab, n_embd))
    t("output_norm.weight", (n_embd,))
    t("output.weight", (n_vocab, n_embd))
    for i in range(n_layer):
        t(f"blk.{i}.attn_norm.weight", (n_embd,))
        for part in ("attn_q", "attn_k", "attn_v", "attn_output"):
            t(f"blk.{i}.{part}.weight", (n_embd, n_embd))
        t(f"blk.{i}.ffn_norm.weight", (n_embd,))
        t(f"blk.{i}.ffn_gate.weight", (n_ff, n_embd))
        t(f"blk.{i}.ffn_down.weight", (n_embd, n_ff))
        t(f"blk.{i}.ffn_up.weight", (n_ff, n_embd))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def _have_binary():
    try:
        vendored.find_binary("llama-quantize", search_path=True)
    except VendorError:
        return False
    return True


needs_binary = pytest.mark.skipif(
    not _have_binary(),
    reason="no vendored or system llama-quantize; run tools/vendor_llama.py")


# ── the output name, which is what makes the output checkable ───────────────────────
@pytest.mark.parametrize(("src", "quant", "want"), [
    ("m-f16.gguf", "Q4_K_M", "m-Q4_K_M.gguf"),
    ("m-F16.gguf", "Q4_K_M", "m-Q4_K_M.gguf"),
    ("model.gguf", "Q8_0", "model-Q8_0.gguf"),
    ("LFM2-abliterated-f16.gguf", "Q4_K_M", "LFM2-abliterated-Q4_K_M.gguf"),
    ("m-Q8_0.gguf", "Q4_K_M", "m-Q4_K_M.gguf"),
])
def test_the_default_output_names_its_own_quantisation(src, quant, want):
    # Not cosmetic: gguf_io.verify checks a file against the quant its NAME claims, so a name
    # that lies is a check that cannot run.
    assert quantise.default_output(src, quant).name == want


def test_the_default_output_stays_beside_the_source(tmp_path):
    out = quantise.default_output(tmp_path / "sub" / "m-f16.gguf", "Q4_K_M")
    assert out.parent == tmp_path / "sub"


# ── the pre-flight, all of it before a long job starts ──────────────────────────────
def test_a_missing_source_says_so(tmp_path):
    with pytest.raises(SystemExit, match="no source GGUF"):
        quantise.preflight(tmp_path / "absent.gguf", tmp_path / "o.gguf", "Q4_K_M",
                           allow_requantize=False, force=False)


def test_a_source_that_is_not_a_gguf_gets_a_sentence_not_a_traceback(tmp_path):
    """Baseline 9: a user is owed plain language, and a GGUFError traceback is not that."""
    bad = tmp_path / "fake.gguf"
    bad.write_text("not a gguf", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        quantise.preflight(bad, tmp_path / "o.gguf", "Q4_K_M",
                           allow_requantize=False, force=False)
    assert "cannot quantise" in str(e.value)
    assert "too short" in str(e.value)


def test_writing_over_the_source_is_refused_before_the_exists_check(tmp_path):
    """Ordering matters: the other way round, only --force reaches this, having already been told
    to pass --force.
    """
    src = _tiny_gguf(tmp_path / "m-f32.gguf")
    with pytest.raises(SystemExit, match="output path is the source path"):
        quantise.preflight(src, src, "Q4_K_M", allow_requantize=False, force=True)


def test_an_existing_output_is_not_silently_overwritten(tmp_path):
    src = _tiny_gguf(tmp_path / "m-f32.gguf")
    out = tmp_path / "m-Q4_K_M.gguf"
    out.write_bytes(b"previous result")
    with pytest.raises(SystemExit, match="Pass --force"):
        quantise.preflight(src, out, "Q4_K_M", allow_requantize=False, force=False)


def test_force_allows_the_overwrite(tmp_path):
    src = _tiny_gguf(tmp_path / "m-f32.gguf")
    out = tmp_path / "m-Q4_K_M.gguf"
    out.write_bytes(b"previous result")
    assert quantise.preflight(src, out, "Q4_K_M", allow_requantize=False,
                              force=True)["file_type"] == "F32"


def test_requantising_is_refused_by_default(tmp_path):
    """Lossy on top of lossy, and the result is indistinguishable from a single-step one."""
    src = _tiny_gguf(tmp_path / "m.gguf", ftype=7)          # declares itself Q8_0
    with pytest.raises(SystemExit) as e:
        quantise.preflight(src, tmp_path / "o.gguf", "Q4_K_M",
                           allow_requantize=False, force=False)
    assert "already a Q8_0" in str(e.value)
    assert "--allow-requantize" in str(e.value)


def test_requantising_is_allowed_when_asked_for_in_writing(tmp_path):
    src = _tiny_gguf(tmp_path / "m.gguf", ftype=7)
    assert quantise.preflight(src, tmp_path / "o.gguf", "Q4_K_M",
                              allow_requantize=True, force=False)["file_type"] == "Q8_0"


@pytest.mark.parametrize("ftype", [0, 1, 32])    # F32, F16, BF16
def test_every_unquantised_source_type_is_accepted(tmp_path, ftype):
    src = _tiny_gguf(tmp_path / f"m{ftype}.gguf", ftype=ftype)
    assert quantise.preflight(src, tmp_path / "o.gguf", "Q4_K_M",
                              allow_requantize=False, force=False)


# ── the whole thing, through the vendored binary ────────────────────────────────────
@needs_binary
def test_a_real_quantisation_produces_a_verified_q4_k_m(tmp_path):
    """THE END-TO-END. Proves the vendored binary is usable, not merely present.

    Vendoring reported success once while producing a binary that could not start, so "the file
    is in the wheel" and "quantisation works" are different claims and this asserts the second.
    """
    src = _tiny_gguf(tmp_path / "tiny-f32.gguf")
    out = tmp_path / "tiny-Q4_K_M.gguf"
    lines = []
    assert quantise.run([str(src), "--type", "Q4_K_M"], log=lines.append) == 0

    head = gguf_io.verify(out, expect_quant="Q4_K_M", expect_arch="llama")
    assert head["file_type"] == "Q4_K_M"
    assert head["tensor_count"] == 21
    # A k-quant is meaningfully smaller than f32, which is the point of running it at all.
    assert out.stat().st_size < src.stat().st_size * 0.4
    assert any("verified" in m for m in lines)


@needs_binary
def test_the_source_is_kept_unless_pruning_is_asked_for(tmp_path):
    src = _tiny_gguf(tmp_path / "tiny-f32.gguf")
    quantise.run([str(src), "--type", "Q4_K_M"], log=lambda _m: None)
    assert src.is_file()


@needs_binary
def test_pruning_removes_the_source_only_after_the_output_verifies(tmp_path):
    """Order matters: deleting the halfway file before the replacement is known good is how one
    bad run costs both files.
    """
    src = _tiny_gguf(tmp_path / "tiny-f32.gguf")
    quantise.run([str(src), "--type", "Q4_K_M", "--prune-source"], log=lambda _m: None)
    assert not src.exists()
    assert (tmp_path / "tiny-Q4_K_M.gguf").is_file()


@needs_binary
def test_keep_source_overrides_pruning(tmp_path):
    src = _tiny_gguf(tmp_path / "tiny-f32.gguf")
    quantise.run([str(src), "--type", "Q4_K_M", "--prune-source", "--keep-source"],
                 log=lambda _m: None)
    assert src.is_file()


@needs_binary
def test_a_quantisation_that_writes_the_wrong_type_is_caught(tmp_path, monkeypatch):
    """The regression guard for believing an exit code over a file.

    The binary is replaced by one that reports success and writes a Q8_0 where a Q4_K_M was
    asked for. Nothing about the process is wrong; the artefact is.
    """
    src = _tiny_gguf(tmp_path / "tiny-f32.gguf")
    out = tmp_path / "tiny-Q4_K_M.gguf"

    def liar(argv, **kw):
        import types
        _tiny_gguf(out, ftype=7)                # a valid GGUF of the WRONG quantisation
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(quantise.subprocess, "run", liar)
    with pytest.raises(SystemExit) as e:
        quantise.run([str(src), "--type", "Q4_K_M"], log=lambda _m: None)
    assert "does not verify" in str(e.value)
    assert out.exists(), "the bad output is kept for inspection, not deleted"


def test_a_failed_quantisation_leaves_no_partial_file(tmp_path, monkeypatch):
    """A partial write is exactly the shape of a real result, which is how a 987 MB fragment of a
    5.16 GB model got recorded as model quality.
    """
    src = _tiny_gguf(tmp_path / "tiny-f32.gguf")
    out = tmp_path / "tiny-Q4_K_M.gguf"

    def fails(argv, **kw):
        import types
        out.write_bytes(b"GGUF" + b"\x00" * 200)     # a partial, plausible-looking file
        return types.SimpleNamespace(returncode=1)

    monkeypatch.setattr(quantise.subprocess, "run", fails)
    monkeypatch.setattr(quantise, "find_binary", lambda *_a, **_k: (tmp_path / "fake", "vendored"))
    with pytest.raises(SystemExit, match="Nothing usable was written"):
        quantise.run([str(src), "--type", "Q4_K_M"], log=lambda _m: None)
    assert not out.exists()


def test_a_missing_binary_explains_both_routes(tmp_path, monkeypatch):
    src = _tiny_gguf(tmp_path / "tiny-f32.gguf")

    def none(*_a, **_k):
        raise VendorError("llama-quantize is not available. vendor_llama.py / PATH / k-quant")

    monkeypatch.setattr(quantise, "find_binary", none)
    with pytest.raises(SystemExit) as e:
        quantise.run([str(src)], log=lambda _m: None)
    assert "vendor_llama.py" in str(e.value)


# ── the imatrix path, added 2026-08-18 ───────────────────────────────────────────
def test_an_imatrix_is_passed_to_the_binary_and_its_calibration_named(tmp_path, monkeypatch):
    """Applying a matrix is half the job; SAYING which one is the half that stops the next
    comparison being confounded. An i1 file and a plain one of the same weights are not
    comparable, and the only place that used to be written down was the filename.
    """
    src = tmp_path / "m-bf16.gguf"
    src.write_bytes(b"GGUF" + b"\0" * 32)
    im = tmp_path / "im.gguf"
    im.write_bytes(b"matrix")
    (tmp_path / ("im.gguf" + ".calibration.json")).write_text(
        '{"calibration": {"corpus": "advbench"}}', encoding="utf-8")

    monkeypatch.setattr(quantise.gguf_io, "verify",
                        lambda *a, **k: {"file_type": "BF16", "tensor_count": 5,
                                         "architecture": "qwen3"})
    monkeypatch.setattr(quantise, "find_binary", lambda _n, **k: ("/x/llama-quantize", "vendored"))
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        Path(argv[-2]).write_bytes(b"GGUF" + b"\0" * 32)
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(quantise.subprocess, "run", fake_run)
    msgs = []
    quantise.run([str(src), str(tmp_path / "o.gguf"), "--type", "Q4_K_M",
                  "--imatrix", str(im)], log=msgs.append)
    assert "--imatrix" in seen["argv"]
    assert any("advbench" in m for m in msgs), "the calibration was applied and not named"


def test_a_matrix_without_a_sidecar_is_applied_and_flagged_as_unknown(tmp_path, monkeypatch):
    src = tmp_path / "m-bf16.gguf"
    src.write_bytes(b"GGUF" + b"\0" * 32)
    im = tmp_path / "orphan.gguf"
    im.write_bytes(b"matrix")
    monkeypatch.setattr(quantise.gguf_io, "verify",
                        lambda *a, **k: {"file_type": "BF16", "tensor_count": 5,
                                         "architecture": "qwen3"})
    monkeypatch.setattr(quantise, "find_binary", lambda _n, **k: ("/x/llama-quantize", "vendored"))
    monkeypatch.setattr(quantise.subprocess, "run",
                        lambda argv, **kw: (Path(argv[-2]).write_bytes(b"GGUF" + b"\0" * 32),
                                            type("R", (), {"returncode": 0})())[1])
    msgs = []
    quantise.run([str(src), str(tmp_path / "o.gguf"), "--type", "Q4_K_M",
                  "--imatrix", str(im)], log=msgs.append)
    assert any("provenance is unknown" in m for m in msgs)


def test_a_missing_imatrix_is_refused_before_the_job(tmp_path, monkeypatch):
    src = tmp_path / "m-bf16.gguf"
    src.write_bytes(b"GGUF" + b"\0" * 32)
    monkeypatch.setattr(quantise.gguf_io, "verify",
                        lambda *a, **k: {"file_type": "BF16", "tensor_count": 5,
                                         "architecture": "qwen3"})
    monkeypatch.setattr(quantise, "find_binary", lambda _n, **k: ("/x/llama-quantize", "vendored"))
    with pytest.raises(SystemExit, match="no importance matrix"):
        quantise.run([str(src), str(tmp_path / "o.gguf"), "--type", "Q4_K_M",
                      "--imatrix", str(tmp_path / "nope.gguf")], log=lambda _m: None)


# ── C-1: the artefact names the toolchain that produced it ───────────────────────
#
# A published figure is measured on a file, and until now the file could not say what made it.
# The confound that motivated this landed next door in the same week: an abliterated arm quantised
# `i1-Q4_K_M` compared against a stock arm quantised plain `Q4_K_M`, two variables in a
# one-variable comparison, caught only because somebody read the filenames.
def test_build_info_reads_what_the_binary_says_about_itself():
    exe, _src = vendored.find_binary("llama-quantize", search_path=True)
    info = quantise.build_info(exe)
    assert info and isinstance(info["build"], int) and info["commit"]


def test_build_info_returns_none_rather_than_guessing(tmp_path):
    """A stub that says nothing about itself yields no field, not an invented one."""
    stub = tmp_path / "quiet"
    stub.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    stub.chmod(0o755)
    assert quantise.build_info(stub) is None


def test_build_info_survives_a_binary_that_does_not_exist(tmp_path):
    assert quantise.build_info(tmp_path / "nope") is None


def test_identity_keeps_the_claim_and_the_report_as_separate_fields():
    """Recording only one would make the interesting case unrepresentable."""
    exe, src = vendored.find_binary("llama-quantize", search_path=True)
    ident = quantise.quantiser_identity(exe, src, log=lambda _m: None)
    assert ident["pinned_tag"], "the pin we believe we vendored is not recorded"
    assert ident["reported_build"]["build"], "what the binary says is not recorded"
    assert len(ident["sha256"]) == 64
    assert "pin_mismatch" not in ident, "this install's pin and binary should agree"


def test_a_binary_that_is_not_the_pinned_one_is_called_out(monkeypatch, tmp_path):
    """The substitution the pins exist to prevent, said out loud rather than left in the JSON.

    This is the whole reason the record carries two fields instead of one.
    """
    said = []
    monkeypatch.setattr(quantise, "pinned_tag", lambda: "b10355")
    monkeypatch.setattr(quantise, "build_info", lambda _e: {"build": 99999, "commit": "deadbeef"})
    stub = tmp_path / "q"
    stub.write_text("x", encoding="utf-8")
    ident = quantise.quantiser_identity(stub, "vendored", log=said.append)
    assert ident["pin_mismatch"] is True
    assert any("not the one this install claims to vendor" in m for m in said)


def test_an_unreadable_pin_manifest_leaves_the_field_empty_rather_than_wrong(monkeypatch):
    from senbonzakura import vendoring
    monkeypatch.setattr(vendoring, "load_manifest",
                        lambda *_a, **_k: (_ for _ in ()).throw(vendoring.VendorError("gone")))
    assert quantise.pinned_tag() is None


@needs_binary
def test_the_sidecar_lands_beside_the_output_and_names_the_toolchain(tmp_path):
    import json
    src = tmp_path / "m.gguf"
    _tiny_gguf(src)
    out = tmp_path / "m-Q4_K_M.gguf"
    assert quantise.run([str(src), str(out), "--type", "Q4_K_M"], log=lambda _m: None) == 0

    sidecar = Path(str(out) + quantise.SIDECAR_SUFFIX)
    assert sidecar.is_file(), "the quantisation recorded nothing about what produced it"
    rec = json.loads(sidecar.read_text(encoding="utf-8"))
    assert rec["schema"] == "senbonzakura-quantisation/1"
    assert rec["quant_type"] == "Q4_K_M"
    assert rec["quantiser"]["tool"] == "llama-quantize"
    assert rec["quantiser"]["reported_build"]["build"] > 0
    assert rec["quantiser"]["sha256"]
    assert rec["imatrix"] is None, "no matrix was applied and the record should say so"
    assert rec["output"]["name"] == out.name and rec["output"]["bytes"] > 0
    assert rec["source"]["name"] == src.name
    assert rec["created"].endswith("+00:00")


@needs_binary
def test_no_sidecar_is_written_when_the_output_does_not_verify(tmp_path, monkeypatch):
    """Provenance for a file that turned out to be wrong is a record of a thing that did not
    happen, so it must not exist.
    """
    src = tmp_path / "m.gguf"
    _tiny_gguf(src)
    out = tmp_path / "m-Q4_K_M.gguf"
    monkeypatch.setattr(gguf_io, "verify",
                        lambda *_a, **_k: (_ for _ in ()).throw(gguf_io.GGUFError("nope")))
    with pytest.raises(SystemExit):
        quantise.run([str(src), str(out), "--type", "Q4_K_M"], log=lambda _m: None)
    assert not Path(str(out) + quantise.SIDECAR_SUFFIX).exists()


@needs_binary
def test_a_sidecar_that_cannot_be_written_degrades_loudly_and_keeps_the_gguf(tmp_path, monkeypatch):
    said = []
    src = tmp_path / "m.gguf"
    _tiny_gguf(src)
    out = tmp_path / "m-Q4_K_M.gguf"
    real_write = Path.write_text

    def _fail_on_sidecar(self, *a, **k):
        if str(self).endswith(quantise.SIDECAR_SUFFIX):
            raise OSError("read-only filesystem")
        return real_write(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", _fail_on_sidecar)
    assert quantise.run([str(src), str(out), "--type", "Q4_K_M"], log=said.append) == 0
    assert out.is_file(), "a provenance failure must not cost the quantisation"
    assert any("provenance is unrecorded" in m for m in said)
