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
import subprocess
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
    """A stand-in for the converter subprocess that writes whatever the test wants it to.

    IT STANDS IN FOR THE CONVERTER AND NOTHING ELSE, which it did not until 2026-09-21.
    `monkeypatch.setattr(convert.subprocess, "run", ...)` reaches the `subprocess` MODULE, not a
    copy of it, so every other caller in the process got this object too. That was invisible
    while `convert.run` made exactly one subprocess call, and it stopped being invisible the
    moment the conversion record started stamping provenance: `crashsafe.git_commit` ran `git
    rev-parse` into a fake expecting `--outfile` and seven tests died inside a helper none of
    them had anything to do with.

    Anything that is not the converter is handed to the real `subprocess.run`, so a test that
    fakes the converter is faking the converter.
    """

    def __init__(self, rc=0, write=None):
        self.rc, self.write, self.calls = rc, write, []
        self._real = subprocess.run

    def __call__(self, argv, **kw):
        if "--outfile" not in argv:
            return self._real(argv, **kw)
        self.calls.append(argv)
        if self.write is not None:
            Path(argv[argv.index("--outfile") + 1]).write_bytes(self.write)
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


# ── the prompt format, which the tensor receipt cannot see ───────────────────────
#
# A GGUF carries the chat template in its own metadata, and llama.cpp, Ollama and vLLM read it
# from there rather than from the checkpoint. Lose it in conversion and the file still loads,
# still generates, and generates against a prompt format the model was never trained on, so the
# failure presents as a bad model rather than a bad export. Every tensor is correct, which is
# exactly why `gguf_io.verify` cannot catch it: it is a statement about the tensors.
#
# Found 2026-09-21 by reading another tool's troubleshooting page, where the same loss is a
# documented, unfixed cause of "gibberish, endless generations or repeated outputs" after export.
def _with_tokenizer(d, body):
    (d / convert.HF_TOKENIZER_CONFIG).write_text(json.dumps(body), encoding="utf-8")
    return d


class TestTheTemplateSurvivedTheExport:
    def test_a_checkpoint_with_a_template_and_an_output_without_one_is_reported(self, tmp_path):
        d = _with_tokenizer(_checkpoint(tmp_path / "m"), {"chat_template": "{{ x }}"})
        said = convert.chat_template_lost(d, _ok_header(metadata={}))
        assert said and convert.GGUF_CHAT_TEMPLATE_KEY in said

    def test_a_template_that_came_through_is_silent(self, tmp_path):
        d = _with_tokenizer(_checkpoint(tmp_path / "m"), {"chat_template": "{{ x }}"})
        header = _ok_header(metadata={convert.GGUF_CHAT_TEMPLATE_KEY: "{{ x }}"})
        assert convert.chat_template_lost(d, header) is None

    def test_a_base_model_with_no_template_is_not_a_finding(self, tmp_path):
        """The check compares against the source rather than asserting a template must exist.

        A base model legitimately carries none, and refusing one would refuse a correct file.
        """
        d = _with_tokenizer(_checkpoint(tmp_path / "m"), {})
        assert convert.chat_template_lost(d, _ok_header(metadata={})) is None

    def test_a_checkpoint_that_cannot_be_read_says_nothing_rather_than_guessing(self, tmp_path):
        """Three distinct unknowns, all of which must stay quiet: absent, unparseable, not a dict.

        `None` is a third answer next to True and False. A check that treated "cannot tell" as
        "lost" would fire on every checkpoint whose tokeniser config it failed to read, which is
        reporting on its own environment rather than on the file.
        """
        d = _checkpoint(tmp_path / "m")
        assert convert.source_chat_template(d) is None
        (d / convert.HF_TOKENIZER_CONFIG).write_text("{ not json", encoding="utf-8")
        assert convert.source_chat_template(d) is None
        _with_tokenizer(d, ["a", "list"])
        assert convert.source_chat_template(d) is None
        assert convert.chat_template_lost(d, _ok_header(metadata={})) is None

    def test_the_template_is_found_where_a_modern_checkpoint_actually_keeps_it(self, tmp_path):
        """THE DEFECT THIS CHECK SHIPPED WITH, found on the ROG against a real checkpoint.

        Transformers used to keep the template under `chat_template` in `tokenizer_config.json`
        and now writes raw Jinja to `chat_template.jinja` beside it, leaving no key in the
        config at all. This function read only the old location, so a current checkpoint
        returned False, which does not mean "cannot tell": it means "definitely none", and
        `chat_template_lost` only speaks when the source is a definite True. The export
        verification was therefore a no-op on exactly the checkpoints anybody would convert.

        LFM2.5-350M is the measured case: 5,487 bytes of template in `chat_template.jinja`, and
        `chat_template` absent from its tokeniser config.
        """
        d = _with_tokenizer(_checkpoint(tmp_path / "m"), {"bos_token": "<s>"})
        (d / convert.HF_CHAT_TEMPLATE_JINJA).write_text(
            "{% for m in messages %}{{ m.content }}{% endfor %}", encoding="utf-8")
        assert convert.source_chat_template(d) is True
        said = convert.chat_template_lost(d, _ok_header(metadata={}))
        assert said and convert.GGUF_CHAT_TEMPLATE_KEY in said

    def test_the_intermediate_json_location_is_read_too(self, tmp_path):
        """`chat_template.json` is the convention between the other two, and processors still
        ship it. Three locations exist in the wild, so all three are looked at.
        """
        d = _with_tokenizer(_checkpoint(tmp_path / "m"), {"bos_token": "<s>"})
        (d / convert.HF_CHAT_TEMPLATE_JSON).write_text(
            json.dumps({"chat_template": "{{ x }}"}), encoding="utf-8")
        assert convert.source_chat_template(d) is True

    def test_an_empty_template_file_is_not_a_template(self, tmp_path):
        """A zero-byte or whitespace-only file is a save that went wrong, not a declaration.

        Treating it as True would make the check fire on every conversion of such a checkpoint
        and blame the exporter for something the checkpoint did.
        """
        d = _with_tokenizer(_checkpoint(tmp_path / "m"), {"bos_token": "<s>"})
        (d / convert.HF_CHAT_TEMPLATE_JINJA).write_text("   \n", encoding="utf-8")
        assert convert.source_chat_template(d) is False

    def test_a_template_file_that_cannot_be_parsed_says_cannot_tell(self, tmp_path):
        """Unreadable is an unknown, and the unknown answer is None rather than False.

        False switches the whole check off silently, which is the failure mode this function
        was just fixed for, so an unparseable sidecar must not land there.
        """
        d = _with_tokenizer(_checkpoint(tmp_path / "m"), {"bos_token": "<s>"})
        (d / convert.HF_CHAT_TEMPLATE_JSON).write_text("{ not json", encoding="utf-8")
        assert convert.source_chat_template(d) is None

    def test_a_checkpoint_with_neither_location_populated_is_still_a_definite_no(self, tmp_path):
        """The base-model case has to keep working, or the fix trades one silent miss for
        a check that fires on every correct file.
        """
        d = _with_tokenizer(_checkpoint(tmp_path / "m"), {"bos_token": "<s>"})
        assert convert.source_chat_template(d) is False

    def test_a_header_with_no_metadata_at_all_does_not_raise(self, tmp_path):
        """Defensive: a header shape without the key must not turn a NOTE into a traceback."""
        d = _with_tokenizer(_checkpoint(tmp_path / "m"), {"chat_template": "{{ x }}"})
        assert convert.chat_template_lost(d, {}) is not None

    @needs_converter
    def test_the_note_actually_reaches_the_log_during_a_conversion(self, tmp_path, monkeypatch):
        """The wiring, not the function. A correct check with no call site is a green suite.

        That has happened twice here, which is why this drives `run` end to end rather than
        calling `chat_template_lost` a sixth time.
        """
        d = _with_tokenizer(_checkpoint(tmp_path / "m"), {"chat_template": "{{ x }}"})
        out = tmp_path / "o.gguf"
        monkeypatch.setattr(convert, "supported_architectures",
                            lambda _s, **k: ({"Qwen3ForCausalLM"}, [], None))
        monkeypatch.setattr(convert.subprocess, "run", _Ran(rc=0, write=b"GGUF"))
        monkeypatch.setattr(convert.gguf_io, "verify", lambda *a, **k: _ok_header(metadata={}))
        lines = []
        convert.run([str(d), str(out)], log=lines.append)
        assert any(convert.GGUF_CHAT_TEMPLATE_KEY in ln for ln in lines), lines

    @needs_converter
    def test_no_note_when_the_template_came_through(self, tmp_path, monkeypatch):
        """The other half, so the wiring test cannot pass by printing the NOTE unconditionally."""
        d = _with_tokenizer(_checkpoint(tmp_path / "m"), {"chat_template": "{{ x }}"})
        monkeypatch.setattr(convert, "supported_architectures",
                            lambda _s, **k: ({"Qwen3ForCausalLM"}, [], None))
        monkeypatch.setattr(convert.subprocess, "run", _Ran(rc=0, write=b"GGUF"))
        monkeypatch.setattr(convert.gguf_io, "verify",
                            lambda *a, **k: _ok_header(metadata={convert.GGUF_CHAT_TEMPLATE_KEY: "t"}))
        lines = []
        convert.run([str(d), str(tmp_path / "o.gguf")], log=lines.append)
        assert not any(convert.GGUF_CHAT_TEMPLATE_KEY in ln for ln in lines), lines


class TestTheConversionRecord:
    """THE RECEIPT, and why the log line was not one.

    `chat_template_lost` produced a sentence on stdout and nothing else, so the one statement
    that a GGUF had lost its prompt format scrolled past an operator and was gone. That is the
    same shape as the `budget_warning` this project wrote into three artefacts and read in none
    of them: a caveat that lives only in a terminal is lost exactly where the number gets quoted
    from. These hold down that a file is written, that it says the true thing in both directions,
    and that it never costs the conversion.
    """

    def _converted(self, tmp_path, monkeypatch, *, source_template, target_template, **kw):
        d = _checkpoint(tmp_path / "m")
        if source_template:
            _with_tokenizer(d, {"chat_template": "{{ x }}"})
        out = tmp_path / "o.gguf"
        monkeypatch.setattr(convert, "supported_architectures",
                            lambda _s, **k: ({"Qwen3ForCausalLM"}, [], None))
        monkeypatch.setattr(convert.subprocess, "run", _Ran(rc=0, write=b"GGUF" + b"\0" * 32))
        meta = {convert.GGUF_CHAT_TEMPLATE_KEY: "t"} if target_template else {}
        monkeypatch.setattr(convert.gguf_io, "verify", lambda *a, **k: _ok_header(metadata=meta))
        rc = convert.run([str(d), str(out), *kw.get("argv", [])], log=lambda _m: None)
        return rc, out

    @needs_converter
    def test_a_conversion_writes_a_record_beside_its_output(self, tmp_path, monkeypatch):
        rc, out = self._converted(tmp_path, monkeypatch,
                                  source_template=True, target_template=True)
        assert rc == 0
        rec = json.loads(convert.record_path(out).read_text(encoding="utf-8"))
        assert rec["record"] == convert.RECORD_KIND
        assert rec["source"]["architecture"] == "Qwen3ForCausalLM"
        assert rec["source"]["declares_chat_template"] is True
        assert rec["target"]["carries_chat_template"] is True
        assert rec["target"]["name"] == "o.gguf"
        assert rec["chat_template_warning"] is None
        assert rec["tool_version"] and rec["provenance"]

    @needs_converter
    def test_the_record_carries_the_warning_the_log_used_to_carry_alone(self, tmp_path,
                                                                       monkeypatch):
        _, out = self._converted(tmp_path, monkeypatch,
                                 source_template=True, target_template=False)
        rec = json.loads(convert.record_path(out).read_text(encoding="utf-8"))
        assert rec["source"]["declares_chat_template"] is True
        assert rec["target"]["carries_chat_template"] is False
        assert convert.GGUF_CHAT_TEMPLATE_KEY in rec["chat_template_warning"]

    @needs_converter
    def test_a_record_is_written_even_when_nothing_went_wrong(self, tmp_path, monkeypatch):
        """A record that appears only on a bad conversion tells a reader nothing about a good
        one: its absence would have to mean either `fine` or `this build is too old to say`.
        """
        _, out = self._converted(tmp_path, monkeypatch,
                                 source_template=False, target_template=False)
        rec = json.loads(convert.record_path(out).read_text(encoding="utf-8"))
        assert rec["chat_template_warning"] is None
        # NONE, NOT FALSE, and the distinction is the one `source_chat_template` was fixed for on
        # 2026-09-21. This checkpoint has no tokeniser config and no template file, so nothing in
        # it has an opinion. A confident False there is indistinguishable from a checked, clean
        # result, and it is what switched the export guard off for a month.
        assert rec["source"]["declares_chat_template"] is None

    @needs_converter
    def test_a_checkpoint_that_says_it_has_no_template_is_recorded_as_saying_so(self, tmp_path,
                                                                                monkeypatch):
        d = _checkpoint(tmp_path / "m")
        _with_tokenizer(d, {"bos_token": "<s>"})
        out = tmp_path / "o.gguf"
        monkeypatch.setattr(convert, "supported_architectures",
                            lambda _s, **k: ({"Qwen3ForCausalLM"}, [], None))
        monkeypatch.setattr(convert.subprocess, "run", _Ran(rc=0, write=b"GGUF"))
        monkeypatch.setattr(convert.gguf_io, "verify", lambda *a, **k: _ok_header(metadata={}))
        convert.run([str(d), str(out)], log=lambda _m: None)
        rec = json.loads(convert.record_path(out).read_text(encoding="utf-8"))
        assert rec["source"]["declares_chat_template"] is False

    @needs_converter
    def test_no_field_at_any_depth_carries_a_name_the_leak_gate_bans(self, tmp_path, monkeypatch):
        """`tools/ci/check_prompt_artefacts.py` refuses a key called `generation`, `prompt`,
        `output`, `text` or `response` anywhere in committed JSON, because that is what a
        retained model reply is called everywhere else here. A record that cannot be committed
        is a record nobody keeps, and this project has already had that exact standoff with its
        own primary artefact.
        """
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "ci"))
        from check_prompt_artefacts import banned_keys_in

        _, out = self._converted(tmp_path, monkeypatch,
                                 source_template=True, target_template=False)
        rec = json.loads(convert.record_path(out).read_text(encoding="utf-8"))
        assert banned_keys_in(rec) == set()

    @needs_converter
    def test_a_record_that_cannot_be_written_does_not_fail_the_conversion(self, tmp_path,
                                                                         monkeypatch, capsys):
        """The GGUF is the product. A conversion that took forty minutes and verified must not
        be reported as a failure because a read-only directory refused a receipt.
        """
        def _boom(*a, **k):
            raise OSError("read-only file system")
        # The RENAME is what is broken, not the write, so the temporary file is created and then
        # has to be cleaned up: this exercises the cleanup as well as the refusal. Patching
        # `write_text` instead would break `_checkpoint` too and test nothing about the record.
        monkeypatch.setattr(Path, "replace", _boom)
        lines = []
        d = _checkpoint(tmp_path / "m")
        out = tmp_path / "o.gguf"
        monkeypatch.setattr(convert, "supported_architectures",
                            lambda _s, **k: ({"Qwen3ForCausalLM"}, [], None))
        monkeypatch.setattr(convert.subprocess, "run", _Ran(rc=0, write=b"GGUF"))
        monkeypatch.setattr(convert.gguf_io, "verify", lambda *a, **k: _ok_header(metadata={}))
        assert convert.run([str(d), str(out)], log=lines.append) == 0
        assert any("could not be written" in ln for ln in lines), lines
        assert not convert.record_path(out).exists()

    @needs_converter
    def test_no_part_file_is_left_behind(self, tmp_path, monkeypatch):
        """Baseline section 2.1: never leave a `.part` behind. A half-written receipt parses as
        far as the reader gets and then stops, which is a file that looks like evidence.
        """
        _, out = self._converted(tmp_path, monkeypatch,
                                 source_template=True, target_template=True)
        assert not list(out.parent.glob("*.part"))

    @needs_converter
    def test_the_checker_reads_the_record_and_finds_the_lost_template(self, tmp_path,
                                                                      monkeypatch):
        """THE WHOLE POINT OF WRITING IT. The check could not be written against a real file
        before, because `convert` produced no file at all.
        """
        from senbonzakura_check import check_document
        from senbonzakura_check.registry import load_checks

        _, out = self._converted(tmp_path, monkeypatch,
                                 source_template=True, target_template=False)
        rec = json.loads(convert.record_path(out).read_text(encoding="utf-8"))
        findings, _ = check_document(rec, load_checks())
        fired = {f.check_id: f.severity for f in findings}
        assert fired.get("a-chat-template-lost-in-conversion") == "withdraws", findings

    @needs_converter
    def test_the_checker_is_quiet_on_a_conversion_that_kept_it(self, tmp_path, monkeypatch):
        from senbonzakura_check import check_document
        from senbonzakura_check.registry import load_checks

        _, out = self._converted(tmp_path, monkeypatch,
                                 source_template=True, target_template=True)
        rec = json.loads(convert.record_path(out).read_text(encoding="utf-8"))
        findings, skipped = check_document(rec, load_checks())
        assert "a-chat-template-lost-in-conversion" not in {f.check_id for f in findings}
        assert "a-chat-template-lost-in-conversion" not in skipped, (
            "the check was skipped on a record that answers its question, which would make its "
            "silence on a real conversion mean nothing")
