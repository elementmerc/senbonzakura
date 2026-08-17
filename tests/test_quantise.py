"""`senbonzakura quantise`, including one genuine end-to-end run through the vendored binary.

The end-to-end test builds a small but STRUCTURALLY REAL llama-architecture GGUF and quantises it
to Q4_K_M, because a k-quant is a per-tensor type mix that llama-quantize chooses from the
architecture metadata: a bag of nameless tensors is not a model it will process. It runs in well
under a second and is skipped where no binary has been vendored, which is any fresh checkout.

Everything else here is a guard, and each guard exists because its absence has cost something:
an exit code that was believed over the file it produced, a partial write left looking like a
result, and a lossy step stacked on a lossy step.
"""
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
