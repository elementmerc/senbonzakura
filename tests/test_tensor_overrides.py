# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Per-tensor precision: asking for it, refusing it when it cannot land, and proving it did.

WHY THIS NEEDS ITS OWN GUARDS

`general.file_type` is a FILE-level label and an override does not touch it. Ask for the output
head at Q8_0, get a file where nothing happened, and every existing check still passes: the name
matches, the architecture matches, the tensor count matches, `verify` is happy. There is no signal
anywhere that the thing you asked for did not occur.

The expensive version of that is a model with TIED embeddings. It has no `output.weight` at all,
so `--output-tensor-type` is accepted by llama-quantize, does nothing, and produces a file
indistinguishable from one where the flag was never passed. This project has already been bitten
once by tied embeddings changing what a conversion produced while every surface reported success.

So there are two guards and they are deliberately at opposite ends. Before the run, refuse an
override that names a tensor the source does not have. After the run, read the finished file back
and confirm the precision is there. The end-to-end test does both against the real binary, because
an override is a claim about a file and only a file can settle it.
"""
from pathlib import Path

import pytest
from artefacts import needs_binary
from test_quantise import _tiny_gguf

from senbonzakura import gguf_io, quantise


class _Args:
    """The subset of the parsed namespace the override helpers read."""

    def __init__(self, **kw):
        self.output_tensor_type = kw.get("output_tensor_type")
        self.token_embedding_type = kw.get("token_embedding_type")
        self.tensor_type = kw.get("tensor_type", [])


# ── parsing NAME=TYPE ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize(("spec", "want"), [
    ("attn_v=Q6_K", ("attn_v", "Q6_K")),
    ("attn_v=q6_k", ("attn_v", "Q6_K")),
    (" attn_v = Q6_K ", ("attn_v", "Q6_K")),
    ("blk.0.ffn_down.weight=F16", ("blk.0.ffn_down.weight", "F16")),
])
def test_a_well_formed_override_parses(spec, want):
    assert quantise.parse_tensor_type(spec) == want


@pytest.mark.parametrize("spec", ["attn_v", "=Q6_K", "  =Q6_K", ""])
def test_an_override_without_a_name_and_a_type_is_refused(spec):
    with pytest.raises(SystemExit, match="NAME=TYPE"):
        quantise.parse_tensor_type(spec)


def test_an_unknown_type_is_refused_with_the_list_of_known_ones():
    """Left to llama-quantize this arrives after the operator has waited for the job to start."""
    with pytest.raises(SystemExit) as e:
        quantise.parse_tensor_type("attn_v=Q9_ULTRA")
    msg = str(e.value)
    assert "Q9_ULTRA" in msg
    assert "Q6_K" in msg, "a refusal has to say what IS allowed"


def test_every_offered_type_is_one_the_reader_knows():
    """A type this tool will pass on and cannot then read back would make the receipt unverifiable."""
    assert set(gguf_io.OVERRIDE_TYPES) <= set(gguf_io.GGML_TYPES.values())


# ── the pre-flight, which is the expensive guard ──────────────────────────────────
def test_an_output_head_override_on_a_tied_model_is_refused(tmp_path):
    """THE CASE THIS EXISTS FOR.

    No output.weight, so the flag is accepted, does nothing, and leaves a file nothing can tell
    apart from one where it was never passed.
    """
    src = _tiny_gguf(tmp_path / "m-f16.gguf", tied=True)
    assert "output.weight" not in {t["name"] for t in gguf_io.read_tensor_info(src)}
    with pytest.raises(SystemExit) as e:
        quantise._preflight_overrides(src, _Args(output_tensor_type="Q8_0"), lambda _m: None)
    msg = str(e.value)
    assert "output.weight" in msg
    assert "tied" in msg, "the message has to name the usual cause or it is not actionable"


def test_the_same_override_is_allowed_when_the_head_is_there(tmp_path):
    src = _tiny_gguf(tmp_path / "m-f16.gguf")
    assert quantise._preflight_overrides(src, _Args(output_tensor_type="Q8_0"),
                                         lambda _m: None) == []


def test_a_pattern_matching_nothing_is_refused(tmp_path):
    src = _tiny_gguf(tmp_path / "m-f16.gguf")
    with pytest.raises(SystemExit, match="matches no tensor"):
        quantise._preflight_overrides(src, _Args(tensor_type=["nonesuch=Q6_K"]),
                                      lambda _m: None)


def test_a_pattern_matching_something_passes_through(tmp_path):
    src = _tiny_gguf(tmp_path / "m-f16.gguf")
    got = quantise._preflight_overrides(src, _Args(tensor_type=["attn_v=Q6_K"]), lambda _m: None)
    assert got == [("attn_v", "Q6_K")]


def test_no_overrides_reads_nothing_from_the_file(tmp_path):
    """The common path must not pay for a feature nobody asked for."""
    missing = tmp_path / "not-there.gguf"
    assert quantise._preflight_overrides(missing, _Args(), lambda _m: None) == []


def test_an_unreadable_tensor_list_warns_rather_than_refusing(tmp_path):
    """Degrade loudly. A file this reader cannot walk may still quantise perfectly well, and
    refusing it would block a working run to protect against a check we could not run.
    """
    bad = tmp_path / "m-f16.gguf"
    bad.write_bytes(b"GGUF" + b"\x00" * 8)
    said = []
    got = quantise._preflight_overrides(bad, _Args(output_tensor_type="Q8_0"), said.append)
    assert got == []
    assert any("unchecked" in s for s in said)


# ── the receipt, after the run ────────────────────────────────────────────────────
def test_a_verified_override_is_recorded_with_the_census(tmp_path):
    src = _tiny_gguf(tmp_path / "m-f16.gguf")
    said = []
    got = quantise._verify_overrides(src, _Args(output_tensor_type="F32"), [], said.append)
    assert got["requested"] == {"output.weight": "F32"}
    assert got["census"]["F32"] > 0
    assert any("verified" in s for s in said)


def test_an_override_that_did_not_land_is_a_loud_failure(tmp_path):
    """An exit code is a statement about a process; this is a statement about a file."""
    src = _tiny_gguf(tmp_path / "m-f16.gguf")     # everything is F32
    with pytest.raises(SystemExit) as e:
        quantise._verify_overrides(src, _Args(output_tensor_type="Q8_0"), [], lambda _m: None)
    msg = str(e.value)
    assert "output.weight was asked for as Q8_0 and is F32" in msg
    assert "general.file_type" in msg, "the message must say why nothing else would have caught it"


def test_a_pattern_override_reports_how_many_tensors_missed(tmp_path):
    src = _tiny_gguf(tmp_path / "m-f16.gguf")
    with pytest.raises(SystemExit) as e:
        quantise._verify_overrides(src, _Args(), [("attn_v", "Q6_K")], lambda _m: None)
    assert "matching 'attn_v'" in str(e.value)


def test_nothing_asked_for_means_nothing_reported(tmp_path):
    src = _tiny_gguf(tmp_path / "m-f16.gguf")
    assert quantise._verify_overrides(src, _Args(), [], lambda _m: None) is None


# ── the real thing ────────────────────────────────────────────────────────────────
@needs_binary
def test_end_to_end_the_head_really_is_kept_at_a_higher_precision(tmp_path):
    """The only test here that proves the flag reaches the binary and changes the bytes.

    Q4_K_M would ordinarily put the output head at Q6_K. Asking for Q8_0 has to show up in the
    finished file, and the rest of the model has to be quantised normally around it: an override
    that accidentally turned into --pure would keep the head and wreck the comparison.
    """
    src = _tiny_gguf(tmp_path / "m-f16.gguf")
    out = tmp_path / "m-Q4_K_M.gguf"
    rc = quantise.run([str(src), str(out), "--type", "Q4_K_M",
                       "--output-tensor-type", "Q8_0"], log=lambda _m: None)
    assert rc == 0

    types = gguf_io.tensor_types(out, ["output.weight", "token_embd.weight"])
    assert types["output.weight"] == "Q8_0", gguf_io.type_census(out)

    census = gguf_io.type_census(out)
    assert any(t.startswith("Q4_K") for t in census), (
        f"the rest of the model was not quantised as asked: {census}")

    sidecar = Path(str(out) + quantise.SIDECAR_SUFFIX)
    record = __import__("json").loads(sidecar.read_text(encoding="utf-8"))
    assert record["tensor_overrides"]["requested"] == {"output.weight": "Q8_0"}
    assert record["tensor_overrides"]["census"]["Q8_0"] >= 1


@needs_binary
def test_end_to_end_a_run_with_no_overrides_records_none(tmp_path):
    src = _tiny_gguf(tmp_path / "m-f16.gguf")
    out = tmp_path / "m-Q4_K_M.gguf"
    assert quantise.run([str(src), str(out), "--type", "Q4_K_M"], log=lambda _m: None) == 0
    record = __import__("json").loads(
        Path(str(out) + quantise.SIDECAR_SUFFIX).read_text(encoding="utf-8"))
    assert record["tensor_overrides"] is None


@needs_binary
def test_end_to_end_the_embedding_and_a_pattern_both_reach_the_binary(tmp_path):
    """The other two flags, and a pattern that matches every layer's value projection.

    Separate from the head test because they build different argv, and an override silently
    dropped while building the command line is exactly as invisible as one that did not land.
    """
    src = _tiny_gguf(tmp_path / "m-f16.gguf")
    out = tmp_path / "m-Q4_K_M.gguf"
    assert quantise.run([str(src), str(out), "--type", "Q4_K_M",
                         "--token-embedding-type", "Q8_0",
                         "--tensor-type", "attn_v=Q8_0"], log=lambda _m: None) == 0

    got = {t["name"]: t["type"] for t in gguf_io.read_tensor_info(out)}
    assert got["token_embd.weight"] == "Q8_0", got
    matched = [n for n in got if "attn_v" in n]
    assert matched, "the fixture has no attn_v tensors, so the pattern proved nothing"
    assert all(got[n] == "Q8_0" for n in matched), {n: got[n] for n in matched}
