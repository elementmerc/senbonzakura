# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The checker reads a published GGUF, and refuses the ones it cannot read.

HOW THE FIXTURES HERE WERE ESTABLISHED, stated plainly for the same reason the adapters package
states it about lm-eval and Inspect: a fixture is a claim about the world and the claim should be
visible.

The byte builder below is **constructed from the GGUF format specification**, which is the weaker
basis. It is not the only basis. The reader was also run against two genuinely published files,
`unsloth/SmolLM2-135M-Instruct-Q4_K_M.gguf` and `...-Q6_K.gguf`, and its tensor census was
compared against `senbonzakura.gguf_io.type_census`, an independently written reader in the main
package. The two agreed exactly on both files:

    Q4_K_M file   {'F32': 61, 'Q4_K': 16, 'Q5_0': 166, 'Q6_K': 14, 'Q8_0': 15}
    Q6_K file     {'F32': 61, 'Q6_K': 30, 'Q8_0': 181}

Those files are 105 MB and 138 MB, so they are not committed and these tests do not download
anything. The agreement is recorded here because it is the evidence that the builder below encodes
the format correctly, and without it these tests would only prove the reader agrees with the test's
own idea of GGUF.
"""
import struct

import pytest
from senbonzakura_check import gguf_read as G
from senbonzakura_check.adapters import detect, normalise
from senbonzakura_check.adapters.gguf_file import GgufAdapter, nominal_type

# GGUF metadata value type numbers, from the specification.
T_UINT32, T_STRING, T_ARRAY = 4, 8, 9


def _str(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def build_gguf(*, version=3, kvs=(), tensors=(), truncate=None, magic=b"GGUF"):
    """GGUF bytes, built from the format specification.

    `kvs` is a sequence of (key, type, payload-bytes). `tensors` is a sequence of
    (name, dims, type_number). `truncate` cuts the result, to make a part-downloaded file.
    """
    out = bytearray(magic)
    out += struct.pack("<I", version)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", len(kvs))
    for key, vtype, payload in kvs:
        out += _str(key) + struct.pack("<I", vtype) + payload
    for name, dims, ttype in tensors:
        out += _str(name) + struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", ttype) + struct.pack("<Q", 0)
    return bytes(out[:truncate] if truncate is not None else out)


def write(tmp_path, data, name="m.gguf"):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def a_real_shaped_file(**over):
    """A file shaped like a published Q4_K_M: a mixed census and a chat template."""
    kvs = [
        ("general.architecture", T_STRING, _str("llama")),
        ("general.name", T_STRING, _str("Test Model")),
        ("general.file_type", T_UINT32, struct.pack("<I", 15)),       # Q4_K_M
        ("general.quantization_version", T_UINT32, struct.pack("<I", 2)),
        (G.CHAT_TEMPLATE_KEY, T_STRING, _str("{{ messages }}")),
    ]
    tensors = [(f"blk.{i}.attn_q.weight", (64, 64), 12) for i in range(3)]   # Q4_K
    tensors += [(f"blk.{i}.ffn_up.weight", (64, 64), 6) for i in range(5)]   # Q5_0
    tensors += [("output_norm.weight", (64,), 0)]                            # F32
    return build_gguf(kvs=kvs, tensors=tensors, **over)


# ── the reader ────────────────────────────────────────────────────────────────────────


def test_a_real_shaped_header_reads(tmp_path):
    h = G.read_header(write(tmp_path, a_real_shaped_file()))
    assert h["version"] == 3
    assert h["architecture"] == "llama"
    assert h["name"] == "Test Model"
    assert h["file_type_number"] == 15
    assert h["file_type"] == "Q4_K_M"
    assert h["quantization_version"] == 2
    assert h["has_chat_template"] is True
    assert h["tensor_count"] == 9
    assert h["tensor_type_census"] == {"F32": 1, "Q4_K": 3, "Q5_0": 5}
    assert h["unknown_tensor_types"] == {}
    assert h["header_bytes_read"] > 0


def test_the_two_type_tables_are_not_the_same_table():
    """The trap the reader is built around, asserted so a merge cannot quietly unify them.

    8 is Q8_0 in the per-tensor enum and Q5_0 in the whole-file enum. A build that looked a
    per-tensor number up in the file-type table would report a plausible wrong precision.
    """
    assert G.GGML_TYPES[8] == "Q8_0"
    assert G.FILE_TYPES[8] == "Q5_0"
    assert G.GGML_TYPES != G.FILE_TYPES


def test_a_file_that_is_not_a_gguf_is_refused(tmp_path):
    p = write(tmp_path, b"\x89PNG\r\n\x1a\n" + b"\x00" * 40, name="not.gguf")
    assert G.looks_like_gguf(p) is False
    with pytest.raises(G.GGUFError, match="does not start with the four bytes GGUF"):
        G.read_header(p)


def test_a_file_too_short_to_hold_a_header_is_refused(tmp_path):
    with pytest.raises(G.GGUFError, match="too short to be a GGUF"):
        G.read_header(write(tmp_path, b"GGUF\x03\x00\x00\x00"))


def test_version_one_is_refused_rather_than_misparsed(tmp_path):
    with pytest.raises(G.GGUFError, match="GGUF version 1"):
        G.read_header(write(tmp_path, a_real_shaped_file(version=1)))


def test_a_truncated_file_names_the_shortfall(tmp_path):
    with pytest.raises(G.GGUFError, match="ends sooner than its own header says"):
        G.read_header(write(tmp_path, a_real_shaped_file(truncate=40)))


def test_a_missing_file_is_a_readable_refusal(tmp_path):
    with pytest.raises(G.GGUFError, match="could not read the file"):
        G.read_header(tmp_path / "absent.gguf")


def test_looks_like_gguf_is_total(tmp_path):
    """It decides which loader a path goes to, so it must not raise on anything."""
    assert G.looks_like_gguf(tmp_path / "absent.gguf") is False
    assert G.looks_like_gguf(tmp_path) is False                  # a directory
    assert G.looks_like_gguf(write(tmp_path, b"", name="empty.gguf")) is False
    assert G.looks_like_gguf(write(tmp_path, a_real_shaped_file())) is True


def test_an_absurd_tensor_count_is_refused(tmp_path):
    data = bytearray(a_real_shaped_file())
    struct.pack_into("<Q", data, 8, G.MAX_TENSORS + 1)
    with pytest.raises(G.GGUFError, match="tensors, past the"):
        G.read_header(write(tmp_path, bytes(data)))


def test_an_absurd_metadata_count_is_refused(tmp_path):
    data = bytearray(a_real_shaped_file())
    struct.pack_into("<Q", data, 16, G.MAX_ELEMENTS + 1)
    with pytest.raises(G.GGUFError, match="metadata entries, which is corrupt"):
        G.read_header(write(tmp_path, bytes(data)))


def test_an_absurd_string_length_is_refused(tmp_path):
    kvs = [("general.name", T_STRING, struct.pack("<Q", G.MAX_ELEMENTS + 1) + b"x")]
    with pytest.raises(G.GGUFError, match=r"past the .* this tool will read"):
        G.read_header(write(tmp_path, build_gguf(kvs=kvs)))


def test_an_unknown_metadata_value_type_is_refused_rather_than_guessed(tmp_path):
    kvs = [("general.name", 99, b"\x00" * 8)]
    with pytest.raises(G.GGUFError, match="metadata value type 99"):
        G.read_header(write(tmp_path, build_gguf(kvs=kvs)))


def test_a_tensor_claiming_too_many_dimensions_is_refused(tmp_path):
    data = build_gguf(tensors=[("t", (1,) * 9, 0)])
    with pytest.raises(G.GGUFError, match="dimensions, which no real model has"):
        G.read_header(write(tmp_path, data))


def test_an_array_value_is_counted_and_not_materialised(tmp_path):
    """A tokeniser vocabulary lives in here, so the reader must walk it without holding it."""
    payload = struct.pack("<I", T_STRING) + struct.pack("<Q", 3)
    payload += _str("a") + _str("b") + _str("c")
    kvs = [("tokenizer.ggml.tokens", T_ARRAY, payload),
           ("general.architecture", T_STRING, _str("llama"))]
    h = G.read_header(write(tmp_path, build_gguf(kvs=kvs)))
    # Walked past it and carried on to the key after it.
    assert h["architecture"] == "llama"


def test_an_unknown_tensor_type_is_named_as_unknown_and_counted(tmp_path):
    data = build_gguf(tensors=[("t1", (8,), 250), ("t2", (8,), 250), ("t3", (8,), 0)])
    h = G.read_header(write(tmp_path, data))
    assert h["unknown_tensor_types"] == {250: 2}
    assert h["tensor_type_census"] == {"F32": 1, "unknown type 250": 2}


def test_a_file_type_number_outside_the_table_reads_as_unknown(tmp_path):
    kvs = [("general.file_type", T_UINT32, struct.pack("<I", 777))]
    h = G.read_header(write(tmp_path, build_gguf(kvs=kvs)))
    assert h["file_type_number"] == 777
    assert h["file_type"] is None, "a number outside the enum must not be rounded to a neighbour"


def test_a_cursor_refuses_a_negative_length():
    with pytest.raises(G.GGUFError, match="negative length"):
        G._Cursor(b"abc").take(-1)


@pytest.mark.parametrize(("name", "expected"), [
    ("model-Q4_K_M.gguf", "Q4_K_M"),
    ("model.Q4_K_S.gguf", "Q4_K_S"),
    ("model_q6_k.gguf", "Q6_K"),
    ("Model-IQ2_XXS.gguf", "IQ2_XXS"),
    ("model.gguf", None),
    ("", None),
])
def test_the_quantisation_a_filename_claims(name, expected):
    assert G.claimed_quant_from_name(name) == expected


def test_the_longest_name_claim_wins():
    """`Q4_K_M` must not be read as `Q4_K` and then reported as disagreeing with itself."""
    assert G.claimed_quant_from_name("x-Q4_K_M.gguf") == "Q4_K_M"
    assert G.claimed_quant_from_name("x-Q3_K_L.gguf") == "Q3_K_L"


# ── the adapter ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("recipe", "expected"), [
    ("Q4_K_M", "Q4_K"), ("Q3_K_L", "Q3_K"), ("Q4_K_S", "Q4_K"), ("Q6_K", "Q6_K"),
    ("Q8_0", "Q8_0"), ("IQ2_XXS", "IQ2_XXS"), ("IQ4_XS", "IQ4_XS"),
    (None, None), ("", None), (17, None),
])
def test_the_nominal_type_a_recipe_is_named_after(recipe, expected):
    assert nominal_type(recipe) == expected


def test_the_iq_family_is_left_alone():
    """Stripping `_XXS` from `IQ2_XXS` would invent a type no file contains."""
    for name in ("IQ2_XXS", "IQ2_XS", "IQ3_S", "IQ1_M"):
        assert nominal_type(name) == name


def _document(tmp_path, data=None, filename="model-Q4_K_M.gguf"):
    from senbonzakura_check.loaders import load_document

    return load_document(write(tmp_path, data or a_real_shaped_file(), name=filename))


def test_the_adapter_owns_a_gguf_document(tmp_path):
    doc = _document(tmp_path)
    assert detect(doc) is GgufAdapter


def test_the_adapter_declines_anything_else():
    assert GgufAdapter.detects({"results": {}, "configs": {}, "versions": {}}) is False
    assert GgufAdapter.detects({}) is False


def test_a_sound_file_normalises_with_no_disagreement(tmp_path):
    n = normalise(_document(tmp_path))
    assert n["harness"] == "gguf file"
    assert n["artefact_kind"] == "gguf"
    assert n["declared_file_type"] == "Q4_K_M"
    assert n["claimed_quant_from_name"] == "Q4_K_M"
    assert n["name_matches_metadata"] is True
    assert n["nominal_tensor_type"] == "Q4_K"
    assert n["dominant_tensor_type"] == "Q5_0"
    assert n["unknown_tensor_type_count"] == 0
    assert n["metrics"] == {}, "a GGUF holds no measurement, and an empty metrics block is correct"
    assert n["imatrix_recorded"] is None


def test_a_renamed_file_is_caught(tmp_path):
    """The real defect: a sound file republished under a name that promises something else."""
    n = normalise(_document(tmp_path, filename="model-Q8_0.gguf"))
    assert n["claimed_quant_from_name"] == "Q8_0"
    assert n["declared_file_type"] == "Q4_K_M"
    assert n["name_matches_metadata"] is False


def test_a_filename_making_no_claim_is_not_a_disagreement(tmp_path):
    n = normalise(_document(tmp_path, filename="model.gguf"))
    assert n["claimed_quant_from_name"] is None
    assert n["name_matches_metadata"] is None, "no claim is not the same as a wrong claim"


def test_a_file_with_no_declared_type_is_not_a_disagreement(tmp_path):
    data = build_gguf(kvs=[("general.architecture", T_STRING, _str("llama"))],
                      tensors=[("t", (8,), 0)])
    n = normalise(_document(tmp_path, data=data, filename="model-Q4_K_M.gguf"))
    assert n["declared_file_type"] is None
    assert n["name_matches_metadata"] is None


def test_a_file_with_no_tensors_reports_no_share_rather_than_zero(tmp_path):
    data = build_gguf(kvs=[("general.file_type", T_UINT32, struct.pack("<I", 15))], tensors=[])
    n = normalise(_document(tmp_path, data=data))
    assert n["nominal_tensor_share"] is None, "nothing to divide by is not 'none of them match'"
    assert n["dominant_tensor_type"] is None
    assert n["dominant_tensor_share"] is None


def test_the_nominal_share_is_a_minority_on_a_realistic_file(tmp_path):
    """The measurement that stopped a check being written. See the adapter's docstring.

    A real `Q4_K_M` carries 5.9% Q4_K tensors and a real `Q6_K` carries 11.0% Q6_K, so a check
    firing on "the nominal type is a minority" would fire on sound files. This asserts the shape
    of that fact so nobody adds the check later without meeting it.
    """
    n = normalise(_document(tmp_path))
    assert n["nominal_tensor_share"] < 0.5
    assert n["dominant_tensor_type"] != n["nominal_tensor_type"]


# ── end to end, through the checks ────────────────────────────────────────────────────


def test_the_name_disagreement_check_fires_end_to_end(tmp_path):
    from senbonzakura_check.cli import inspect_file
    from senbonzakura_check.registry import load_checks

    p = write(tmp_path, a_real_shaped_file(), name="model-Q8_0.gguf")
    findings, _skipped, problem = inspect_file(p, load_checks())
    assert problem is None
    assert "a-file-whose-name-disagreeing" not in [f.check_id for f in findings]
    assert "a-file-whose-name-disagrees-with-its-own-metadata" in [f.check_id for f in findings]


def test_a_sound_file_produces_no_findings(tmp_path):
    from senbonzakura_check.cli import inspect_file
    from senbonzakura_check.registry import load_checks

    p = write(tmp_path, a_real_shaped_file(), name="model-Q4_K_M.gguf")
    findings, _skipped, problem = inspect_file(p, load_checks())
    assert problem is None
    assert findings == []


def test_the_unknown_type_check_fires(tmp_path):
    from senbonzakura_check.cli import inspect_file
    from senbonzakura_check.registry import load_checks

    kvs = [("general.file_type", T_UINT32, struct.pack("<I", 15))]
    data = build_gguf(kvs=kvs, tensors=[("t", (8,), 250), ("u", (8,), 12)])
    p = write(tmp_path, data, name="model-Q4_K_M.gguf")
    findings, _skipped, problem = inspect_file(p, load_checks())
    assert problem is None
    assert "a-file-using-tensor-types-this-build-cannot-name" in [f.check_id for f in findings]


def test_a_corrupt_gguf_reads_as_unchecked_and_never_as_clean(tmp_path):
    from senbonzakura_check.cli import inspect_file
    from senbonzakura_check.registry import load_checks

    p = write(tmp_path, a_real_shaped_file(truncate=40))
    findings, _skipped, problem = inspect_file(p, load_checks())
    assert findings == []
    assert problem is not None, "a file nobody could parse must not report as clean"
    assert "could not read it as an artefact" in str(problem)
