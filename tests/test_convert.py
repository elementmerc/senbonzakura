# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`senbonzakura convert`: safetensors to GGUF, and the pre-flight that stops a long job early.

The conversion itself is upstream code and is not what these test. What they hold down is the
boundary around it: the checks that run before a job measured in minutes starts, and the read-back
that distinguishes "the process exited 0" from "the file is what was asked for". Those two came
apart once already on a 987 MB fragment of a 5.16 GB GGUF that loaded, served, and answered
nonsense that was recorded as model quality.

The sharpest test here is `test_a_module_that_failed_to_import_is_not_support`, which is the defect
that actually shipped: the converter's supported-architecture list is a static registry, so an
architecture whose module raises on import stays on the list. Support claimed, support absent, exit
code zero.
"""
import json
from pathlib import Path

import pytest
from artefacts import needs_converter

from senbonzakura import convert
from senbonzakura.convert import ConvertError


def _checkpoint(d, *, arch="Qwen3ForCausalLM", weights=True, extra=None):
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"architectures": [arch], "model_type": "qwen3", "hidden_size": 64}
    cfg.update(extra or {})
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    if weights:
        (d / "model.safetensors").write_bytes(b"\0" * 2048)
    return d


# ── reading the checkpoint ───────────────────────────────────────────────────────
def test_a_missing_directory_says_so(tmp_path):
    with pytest.raises(ConvertError, match="no model directory"):
        convert.read_config(tmp_path / "nope")


def test_a_gguf_mistaken_for_a_checkpoint_is_pointed_at_the_right_command(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "model.gguf").write_bytes(b"GGUF")
    with pytest.raises(ConvertError, match="quantise"):
        convert.read_config(d)


def test_unreadable_config_is_a_plain_failure(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ConvertError, match="not readable JSON"):
        convert.read_config(d)


# ── the pre-flight ───────────────────────────────────────────────────────────────
def test_a_config_only_directory_is_refused(tmp_path):
    """Exactly what a Hub cache leaves behind when only metadata was ever fetched, and it is a
    directory that looks complete. Verified on this machine: the cached LFM2.5-8B-A1B entry is one
    config.json, and a picker that trusted the cache would have offered it as a ready 17 GB model.
    """
    d = _checkpoint(tmp_path / "m", weights=False)
    with pytest.raises(ConvertError, match="converts to nothing"):
        convert.preflight(d, tmp_path / "o.gguf", force=False, skip_arch_check=True)


def test_an_already_quantised_checkpoint_is_refused(tmp_path):
    d = _checkpoint(tmp_path / "m", extra={"quantization_config": {"quant_method": "gptq"}})
    with pytest.raises(ConvertError, match="already quantised"):
        convert.preflight(d, tmp_path / "o.gguf", force=False, skip_arch_check=True)


def test_a_config_without_an_architecture_is_refused(tmp_path):
    d = _checkpoint(tmp_path / "m")
    (d / "config.json").write_text(json.dumps({"model_type": "x"}), encoding="utf-8")
    with pytest.raises(ConvertError, match="names no architecture"):
        convert.preflight(d, tmp_path / "o.gguf", force=False, skip_arch_check=True)


def test_writing_over_the_model_directory_is_refused(tmp_path):
    d = _checkpoint(tmp_path / "m")
    with pytest.raises(ConvertError, match="output path is the model directory"):
        convert.preflight(d, d, force=False, skip_arch_check=True)


@needs_converter
def test_an_existing_output_needs_force(tmp_path):
    d = _checkpoint(tmp_path / "m")
    out = tmp_path / "o.gguf"
    out.write_bytes(b"x")
    with pytest.raises(ConvertError, match="Pass --force"):
        convert.preflight(d, out, force=False, skip_arch_check=True)
    convert.preflight(d, out, force=True, skip_arch_check=True)


@needs_converter
def test_not_enough_disk_is_caught_before_the_job_not_during(tmp_path, monkeypatch):
    d = _checkpoint(tmp_path / "m")
    monkeypatch.setattr(convert, "free_bytes_for", lambda _p: 1)
    with pytest.raises(ConvertError, match="free space"):
        convert.preflight(d, tmp_path / "o.gguf", force=False, skip_arch_check=True)


# ── the architecture check, and the failure it exists for ────────────────────────
@needs_converter
def test_an_unsupported_architecture_is_refused_before_the_weights_are_read(tmp_path, monkeypatch):
    d = _checkpoint(tmp_path / "m", arch="NoSuchForCausalLM")
    monkeypatch.setattr(convert, "supported_architectures",
                        lambda _s, **k: ({"Qwen3ForCausalLM", "Lfm2MoeForCausalLM"}, [], None))
    with pytest.raises(ConvertError, match="does not support NoSuchForCausalLM"):
        convert.preflight(d, tmp_path / "o.gguf", force=False, skip_arch_check=False)


@needs_converter
def test_a_supported_architecture_passes(tmp_path, monkeypatch):
    d = _checkpoint(tmp_path / "m")
    monkeypatch.setattr(convert, "supported_architectures",
                        lambda _s, **k: ({"Qwen3ForCausalLM"}, [], None))
    got = convert.preflight(d, tmp_path / "o.gguf", force=False, skip_arch_check=False)
    assert got["architecture"] == "Qwen3ForCausalLM"
    assert got["shards"] == 1


@needs_converter
def test_a_module_that_failed_to_import_is_not_support(tmp_path, monkeypatch):
    """THE defect this check exists for, and it shipped.

    The converter's supported list comes from a static registry, so an architecture whose module
    raises on import is still advertised. That is what a mismatched `gguf` package produced: PyPI
    gguf and the pinned llama.cpp's own gguf-py both call themselves 0.19.0, the PyPI one lacks
    constants the converter uses, `conversion/lfm2.py` raised, and `Lfm2MoeForCausalLM` stayed on
    the list. A run would have loaded the weights and then died.
    """
    d = _checkpoint(tmp_path / "m", arch="Lfm2MoeForCausalLM")
    monkeypatch.setattr(convert, "supported_architectures",
                        lambda _s, **k: ({"Lfm2MoeForCausalLM"}, ["lfm2"], None))
    with pytest.raises(ConvertError) as e:
        convert.preflight(d, tmp_path / "o.gguf", force=False, skip_arch_check=False)
    assert "advertised and absent" in str(e.value)
    assert "vendor_llama.py" in str(e.value)


@needs_converter
def test_the_arch_check_can_be_skipped_without_skipping_the_rest(tmp_path, monkeypatch):
    d = _checkpoint(tmp_path / "m", arch="SomethingExotic")
    called = []
    monkeypatch.setattr(convert, "supported_architectures",
                        lambda _s, **k: called.append(1) or (set(), []))
    convert.preflight(d, tmp_path / "o.gguf", force=False, skip_arch_check=True)
    assert not called, "the architecture check ran despite being skipped"


def test_supported_architectures_parses_names_and_broken_modules():
    """Parsed from the converter's own output rather than kept as a list here, because a list here
    would go stale at the next pin bump and would then refuse models the converter accepts.
    """
    sample = ("Failed to load model module lfm2: no attribute 'X'\n"
              "Supported models:\n  - Qwen3ForCausalLM\n  - Lfm2MoeForCausalLM\n")

    class _R:
        returncode = 0
        stdout = sample.encode()
        stderr = b""

    import senbonzakura.convert as c
    real = c.subprocess.run
    c.subprocess.run = lambda *a, **k: _R()
    try:
        names, broken, _died = c.supported_architectures("x")
    finally:
        c.subprocess.run = real
    assert names == {"Qwen3ForCausalLM", "Lfm2MoeForCausalLM"}
    assert broken == ["lfm2"]


# ── naming ───────────────────────────────────────────────────────────────────────
def test_the_default_output_sits_beside_the_model_and_names_its_precision(tmp_path):
    d = _checkpoint(tmp_path / "lfm2-brain")
    out = convert.default_output(d, "bf16")
    assert out.name == "lfm2-brain-bf16.gguf"
    assert out.parent == d.parent


def test_every_offered_outtype_can_be_verified_afterwards():
    """A type the header reader cannot name is a type a conversion could silently get wrong."""
    for t in convert.OUT_TYPES:
        assert t == "auto" or t in convert.HEADER_TYPE, f"{t} has no header name to check against"


# ── run(): the parts that delete files and chain to another command ──────────────
class _Ran:
    """A stand-in for the converter subprocess that writes whatever the test wants it to."""

    def __init__(self, rc=0, write=None):
        self.rc, self.write, self.calls = rc, write, []

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        if self.write is not None:
            out = Path(argv[argv.index("--outfile") + 1])
            out.write_bytes(self.write)
        return type("R", (), {"returncode": self.rc})()


def _ok_header(**over):
    h = {"file_type": "BF16", "tensor_count": 25, "architecture": "qwen3"}
    h.update(over)
    return h


@needs_converter
def test_a_failed_conversion_removes_the_partial_file(tmp_path, monkeypatch, capsys):
    """A converter that dies halfway leaves a file the same shape as a real one. Leaving it is how
    a 987 MB fragment of a 5.16 GB GGUF got served and scored as model quality.
    """
    d = _checkpoint(tmp_path / "m")
    out = tmp_path / "o.gguf"
    monkeypatch.setattr(convert, "supported_architectures", lambda _s, **k: ({"Qwen3ForCausalLM"}, [], None))
    monkeypatch.setattr(convert.subprocess, "run", _Ran(rc=1, write=b"partial"))
    with pytest.raises(SystemExit, match="exited 1"):
        convert.run([str(d), str(out)], log=lambda _m: None)
    assert not out.exists(), "the partial output was left behind"


@needs_converter
def test_an_output_that_does_not_verify_is_kept_for_inspection(tmp_path, monkeypatch):
    """The opposite decision from the failure above, deliberately. A process that reported success
    and produced something unverifiable is a bug worth looking at, so the evidence stays.
    """
    d = _checkpoint(tmp_path / "m")
    out = tmp_path / "o.gguf"
    monkeypatch.setattr(convert, "supported_architectures", lambda _s, **k: ({"Qwen3ForCausalLM"}, [], None))
    monkeypatch.setattr(convert.subprocess, "run", _Ran(rc=0, write=b"not a gguf"))

    def _boom(*a, **k):
        raise convert.gguf_io.GGUFError("bad magic")
    monkeypatch.setattr(convert.gguf_io, "verify", _boom)

    with pytest.raises(SystemExit, match="does not verify"):
        convert.run([str(d), str(out)], log=lambda _m: None)
    assert out.exists(), "the unverifiable output was deleted instead of kept for inspection"


@needs_converter
def test_a_successful_conversion_verifies_and_reports(tmp_path, monkeypatch):
    d = _checkpoint(tmp_path / "m")
    out = tmp_path / "o.gguf"
    monkeypatch.setattr(convert, "supported_architectures", lambda _s, **k: ({"Qwen3ForCausalLM"}, [], None))
    ran = _Ran(rc=0, write=b"GGUF" + b"\0" * 64)
    monkeypatch.setattr(convert.subprocess, "run", ran)
    monkeypatch.setattr(convert.gguf_io, "verify", lambda *a, **k: _ok_header())
    msgs = []
    assert convert.run([str(d), str(out)], log=msgs.append) == 0
    assert any("verified" in m for m in msgs)
    # The precision asked for must be the precision asserted on the way out.
    assert "--outtype" in ran.calls[0] and "bf16" in ran.calls[0]


@needs_converter
def test_the_requested_precision_is_what_gets_checked(tmp_path, monkeypatch):
    d = _checkpoint(tmp_path / "m")
    out = tmp_path / "o.gguf"
    monkeypatch.setattr(convert, "supported_architectures", lambda _s, **k: ({"Qwen3ForCausalLM"}, [], None))
    monkeypatch.setattr(convert.subprocess, "run", _Ran(rc=0, write=b"GGUF"))
    seen = {}

    def _verify(path, *, expect_quant=None, **k):
        seen["expect"] = expect_quant
        return _ok_header(file_type="F16")
    monkeypatch.setattr(convert.gguf_io, "verify", _verify)
    convert.run([str(d), str(out), "--outtype", "f16"], log=lambda _m: None)
    assert seen["expect"] == "F16"


@needs_converter
def test_auto_precision_asserts_nothing_because_there_is_nothing_to_assert(tmp_path, monkeypatch):
    d = _checkpoint(tmp_path / "m")
    out = tmp_path / "o.gguf"
    monkeypatch.setattr(convert, "supported_architectures", lambda _s, **k: ({"Qwen3ForCausalLM"}, [], None))
    monkeypatch.setattr(convert.subprocess, "run", _Ran(rc=0, write=b"GGUF"))
    seen = {}

    def _verify(path, *, expect_quant=None, **k):
        seen["expect"] = expect_quant
        return _ok_header()
    monkeypatch.setattr(convert.gguf_io, "verify", _verify)
    convert.run([str(d), str(out), "--outtype", "auto"], log=lambda _m: None)
    assert seen["expect"] is False, "auto claimed a precision it cannot know in advance"


@needs_converter
def test_quantise_is_chained_and_the_intermediate_is_pruned_by_default(tmp_path, monkeypatch):
    d = _checkpoint(tmp_path / "m")
    out = tmp_path / "o.gguf"
    monkeypatch.setattr(convert, "supported_architectures", lambda _s, **k: ({"Qwen3ForCausalLM"}, [], None))
    monkeypatch.setattr(convert.subprocess, "run", _Ran(rc=0, write=b"GGUF"))
    monkeypatch.setattr(convert.gguf_io, "verify", lambda *a, **k: _ok_header())
    from senbonzakura import quantise
    seen = {}

    def _q(argv, log=print):
        seen["argv"] = argv
        return 0
    monkeypatch.setattr(quantise, "run", _q)
    assert convert.run([str(d), str(out), "--quantise", "Q4_K_M"], log=lambda _m: None) == 0
    assert "--prune-source" in seen["argv"], "the full-precision intermediate was not cleaned up"
    assert "Q4_K_M" in seen["argv"]


@needs_converter
def test_keeping_the_intermediate_is_honoured(tmp_path, monkeypatch):
    d = _checkpoint(tmp_path / "m")
    out = tmp_path / "o.gguf"
    monkeypatch.setattr(convert, "supported_architectures", lambda _s, **k: ({"Qwen3ForCausalLM"}, [], None))
    monkeypatch.setattr(convert.subprocess, "run", _Ran(rc=0, write=b"GGUF"))
    monkeypatch.setattr(convert.gguf_io, "verify", lambda *a, **k: _ok_header())
    from senbonzakura import quantise
    seen = {}
    monkeypatch.setattr(quantise, "run", lambda argv, log=print: seen.update(argv=argv) or 0)
    convert.run([str(d), str(out), "--quantise", "Q4_K_M", "--keep-intermediate"],
                log=lambda _m: None)
    assert "--prune-source" not in seen["argv"]


@needs_converter
def test_a_failing_quantise_step_propagates_its_code(tmp_path, monkeypatch):
    d = _checkpoint(tmp_path / "m")
    out = tmp_path / "o.gguf"
    monkeypatch.setattr(convert, "supported_architectures", lambda _s, **k: ({"Qwen3ForCausalLM"}, [], None))
    monkeypatch.setattr(convert.subprocess, "run", _Ran(rc=0, write=b"GGUF"))
    monkeypatch.setattr(convert.gguf_io, "verify", lambda *a, **k: _ok_header())
    from senbonzakura import quantise
    monkeypatch.setattr(quantise, "run", lambda argv, log=print: 3)
    assert convert.run([str(d), str(out), "--quantise", "Q4_K_M"], log=lambda _m: None) == 3


@needs_converter
def test_use_temp_file_reaches_the_converter(tmp_path, monkeypatch):
    """The flag that makes a model larger than memory convertible at all."""
    d = _checkpoint(tmp_path / "m")
    out = tmp_path / "o.gguf"
    monkeypatch.setattr(convert, "supported_architectures", lambda _s, **k: ({"Qwen3ForCausalLM"}, [], None))
    ran = _Ran(rc=0, write=b"GGUF")
    monkeypatch.setattr(convert.subprocess, "run", ran)
    monkeypatch.setattr(convert.gguf_io, "verify", lambda *a, **k: _ok_header())
    convert.run([str(d), str(out), "--use-temp-file"], log=lambda _m: None)
    assert "--use-temp-file" in ran.calls[0]


def test_a_missing_vendored_converter_is_a_plain_failure(tmp_path, monkeypatch):
    d = _checkpoint(tmp_path / "m")
    monkeypatch.setattr(convert, "find_script",
                        lambda _n: (_ for _ in ()).throw(convert.VendorError("not fetched")))
    with pytest.raises(SystemExit, match="not fetched"):
        convert.run([str(d), str(tmp_path / "o.gguf")], log=lambda _m: None)
