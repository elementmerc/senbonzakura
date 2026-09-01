# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The header reader, checked against bytes rather than against a mock.

Every fixture here is a real GGUF header assembled field by field to the format specification. A
test that stubs the parse would pass against a parser that reads the format wrongly, which is the
one thing this module exists to get right.

The case that motivates the module: a COMPLETE file of exactly the promised length that is the
wrong quantisation. It loads, it serves, and it changes every speed and memory number it appears
in. Byte counts cannot catch it and neither can a checksum of a file you have never seen before.
"""
import struct

import pytest

from senbonzakura import gguf_io
from senbonzakura.gguf_io import GGUFError

U32, STRING = 4, 8


def _kv_string(key, value):
    kb, vb = key.encode(), value.encode()
    return (struct.pack("<Q", len(kb)) + kb + struct.pack("<I", STRING)
            + struct.pack("<Q", len(vb)) + vb)


def _kv_u32(key, value):
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", U32) + struct.pack("<I", value)


def _gguf(tmp_path, name="model-Q4_K_M.gguf", *, version=3, tensors=291,
          arch="lfm2", ftype=15, extra=b"", kv_count=None):
    """A real GGUF header on disk. `ftype=None` omits general.file_type entirely."""
    kvs = [_kv_string("general.architecture", arch)] if arch is not None else []
    if ftype is not None:
        kvs.append(_kv_u32("general.file_type", ftype))
    body = b"".join(kvs) + extra
    n_kv = kv_count if kv_count is not None else len(kvs) + (1 if extra else 0)
    head = (gguf_io.MAGIC + struct.pack("<I", version)
            + struct.pack("<Q", tensors) + struct.pack("<Q", n_kv) + body)
    p = tmp_path / name
    p.write_bytes(head + b"\x00" * 512)          # tensor data the reader must not need
    return p


# ── reading a real header ──────────────────────────────────────────────────────────
def test_a_real_header_parses(tmp_path):
    h = gguf_io.read_header(_gguf(tmp_path))
    assert h["version"] == 3
    assert h["tensor_count"] == 291
    assert h["architecture"] == "lfm2"
    assert h["file_type"] == "Q4_K_M"
    assert h["file_type_id"] == 15


def test_only_the_head_is_read(tmp_path):
    """A 5 GB model must not be read into memory to answer 'what are you'."""
    p = _gguf(tmp_path)
    with p.open("ab") as f:
        f.write(b"\x00" * (4 * 1024 * 1024))
    h = gguf_io.read_header(p, max_bytes=64 * 1024)
    assert h["file_type"] == "Q4_K_M"


def test_a_read_window_too_small_is_not_reported_as_a_damaged_file(tmp_path):
    """Found by parsing a REAL llama.cpp file rather than a fixture.

    A genuine LFM2.5-1.2B Q4_K_M header is over a megabyte: 65,536 tokeniser tokens, their types
    and 63,683 merges all live in the metadata. Every synthetic fixture in this file is a few
    hundred bytes, so none of them could ever have caught this. Running out of buffer is the same
    event whether the FILE ended or the READ did, and calling a healthy 5 GB model corrupt because
    the window was too small is the wrong accusation in the more expensive direction.
    """
    big = _gguf(tmp_path, extra=b"".join(
        _kv_string(f"tokenizer.ggml.tokens.{i}", "x" * 200) for i in range(400)), kv_count=402)
    with pytest.raises(GGUFError) as e:
        gguf_io.read_header(big, max_bytes=4096)
    msg = str(e.value)
    assert "limit on the read" in msg
    assert "NOT" in msg and "damaged" in msg
    assert "truncated" not in msg          # the accusation it must NOT make
    # The same file parses when it is allowed to read enough.
    assert gguf_io.read_header(big)["file_type"] == "Q4_K_M"


def test_every_known_file_type_round_trips(tmp_path):
    for fid, name in gguf_io.FILE_TYPES.items():
        h = gguf_io.read_header(_gguf(tmp_path, name=f"m{fid}.gguf", ftype=fid))
        assert h["file_type"] == name


# ── the things that are not a GGUF ─────────────────────────────────────────────────
def test_an_html_error_page_is_named_as_one(tmp_path):
    """A failed download saves at exactly the length the server promised."""
    p = tmp_path / "model-Q4_K_M.gguf"
    p.write_bytes(b"<!DOCTYPE html><html><body>404 Not Found</body></html>")
    with pytest.raises(GGUFError) as e:
        gguf_io.read_header(p)
    assert "HTML page" in str(e.value)
    assert "size check alone will not catch this" in str(e.value)


def test_an_empty_file_says_so(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"")
    with pytest.raises(GGUFError, match="too short"):
        gguf_io.read_header(p)


def test_a_missing_file_is_a_readable_error(tmp_path):
    with pytest.raises(GGUFError, match="could not read"):
        gguf_io.read_header(tmp_path / "absent.gguf")


def test_a_truncated_header_is_refused_not_half_parsed(tmp_path):
    """Slicing returns a short result silently; the cursor must not."""
    full = _gguf(tmp_path).read_bytes()
    p = tmp_path / "cut.gguf"
    p.write_bytes(full[:30])                      # past the magic, inside the metadata
    with pytest.raises(GGUFError, match=r"truncated|ends after"):
        gguf_io.read_header(p)


def test_an_unsupported_version_stops_rather_than_misparses(tmp_path):
    with pytest.raises(GGUFError) as e:
        gguf_io.read_header(_gguf(tmp_path, version=1))
    assert "version 1" in str(e.value)
    assert "misread every field" in str(e.value)


def test_a_corrupt_length_cannot_exhaust_memory(tmp_path):
    """These numbers come off disk. read(n) on garbage is how a header read becomes an OOM."""
    head = (gguf_io.MAGIC + struct.pack("<I", 3) + struct.pack("<Q", 1)
            + struct.pack("<Q", 1) + struct.pack("<Q", 2**60))     # absurd key length
    p = tmp_path / "m.gguf"
    p.write_bytes(head)
    with pytest.raises(GGUFError, match="corrupt rather than large"):
        gguf_io.read_header(p)


def test_an_unknown_value_type_is_refused(tmp_path):
    kb = b"weird"
    extra = struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 99)
    with pytest.raises(GGUFError, match="unknown GGUF metadata value type"):
        gguf_io.read_header(_gguf(tmp_path, arch=None, ftype=None, extra=extra, kv_count=1))


# ── what the filename claims ───────────────────────────────────────────────────────
@pytest.mark.parametrize(("name", "want"), [
    ("model-Q4_K_M.gguf", "Q4_K_M"),
    ("LFM2.5-1.2B-abliterated-Q4_K_M.gguf", "Q4_K_M"),
    ("lower-case-q4_k_m.gguf", "Q4_K_M"),
    ("model-Q8_0.gguf", "Q8_0"),
    ("model-IQ4_XS.gguf", "IQ4_XS"),
    ("model.gguf", None),
    ("no-quant-here.gguf", None),
])
def test_the_filename_claim_is_read_longest_first(name, want):
    # Q4_K_M must not be read as Q4_K and then reported as a mismatch against itself.
    assert gguf_io.claimed_quant(name) == want


# ── the check the module exists for ────────────────────────────────────────────────
def test_a_matching_file_passes(tmp_path):
    assert gguf_io.verify(_gguf(tmp_path))["file_type"] == "Q4_K_M"


def test_a_q8_wearing_a_q4_name_is_caught(tmp_path):
    """THE CASE. Complete, valid, loads fine, wrong. Nothing else catches it."""
    p = _gguf(tmp_path, name="model-Q4_K_M.gguf", ftype=7)     # actually Q8_0
    with pytest.raises(GGUFError) as e:
        gguf_io.verify(p)
    msg = str(e.value)
    assert "is a Q8_0 file but its name says Q4_K_M" in msg
    assert "it will load" in msg                                # says why it matters


def test_a_file_claiming_nothing_is_not_failed(tmp_path):
    """`model.gguf` is a legitimate name and must not be treated as a mismatch."""
    assert gguf_io.verify(_gguf(tmp_path, name="model.gguf"))["file_type"] == "Q4_K_M"


def test_an_explicit_expectation_beats_the_filename(tmp_path):
    with pytest.raises(GGUFError, match="name says Q8_0"):
        gguf_io.verify(_gguf(tmp_path, name="model-Q4_K_M.gguf"), expect_quant="Q8_0")


def test_the_quant_check_can_be_skipped_explicitly(tmp_path):
    p = _gguf(tmp_path, name="model-Q4_K_M.gguf", ftype=7)
    assert gguf_io.verify(p, expect_quant=False)["file_type"] == "Q8_0"


def test_an_unreadable_file_type_is_not_read_as_agreement(tmp_path):
    """Absent metadata must not pass a check it never made."""
    p = _gguf(tmp_path, name="model-Q4_K_M.gguf", ftype=None)
    with pytest.raises(GGUFError, match="cannot be checked"):
        gguf_io.verify(p)


def test_an_unknown_file_type_id_is_reported_as_unknown(tmp_path):
    # 5 is a removed format and is not reused. Guessing at it would be worse than refusing.
    p = _gguf(tmp_path, name="model-Q4_K_M.gguf", ftype=5)
    with pytest.raises(GGUFError, match=r"no readable general\.file_type"):
        gguf_io.verify(p)


def test_the_wrong_architecture_is_caught(tmp_path):
    with pytest.raises(GGUFError, match="different model"):
        gguf_io.verify(_gguf(tmp_path, arch="qwen3"), expect_arch="lfm2")


def test_a_container_with_no_tensors_is_refused(tmp_path):
    """A valid header with no weights loads and answers nonsense."""
    with pytest.raises(GGUFError, match="fewer than the"):
        gguf_io.verify(_gguf(tmp_path, tensors=0))
