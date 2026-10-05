# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Enough of the GGUF header to say what a published file actually contains.

WHY THIS IS A SECOND READER AND NOT AN IMPORT, WHICH IS THE DECISION THIS FILE RESTS ON

`senbonzakura/gguf_io.py` already reads GGUF headers and imports no torch, so importing it would
be the obvious move. It is not available to us. `checker/pyproject.toml` declares
`dependencies = []`, and the whole point of the `senbonzakura-check` distribution is that
`pip install senbonzakura-check` pulls nothing: no torch, no CUDA, no model, no corpus. A user who
installs only the checker does not have `senbonzakura` on the path at all, so an import across the
boundary would be an `ImportError` on the machine of exactly the person this subpackage exists to
serve.

Vendoring a copy was the other option and was rejected: `one-home.manifest` would need a `pair`
entry requiring the two files to hash-match for ever, across two distributions that ship
separately and on different schedules, and the first divergence would be a failing commit hook
rather than a caught bug.

So this is a deliberate minimal re-implementation, and the justification is the one the adapters
package already makes for reading artefacts instead of importing harnesses: a file format changes
far more slowly than an API, and the format is versioned by the tool that writes it. What it is
NOT is a general GGUF library. It reads the header and stops.

THE TRAP THIS FILE IS BUILT AROUND, AND IT IS NOT HYPOTHETICAL

**`general.file_type` and the per-tensor type enum are different numberings, and 8 means Q8_0 in
one and Q5_0 in the other.** Looking a number up in the wrong table yields a plausible, wrong
answer with no error, which is the exact shape of defect this checker exists to catch in other
people's work. The two tables below are therefore separate, named for what they decode, and
neither is ever consulted for the other's numbers.

Both tables are transcribed from `senbonzakura/gguf_io.py`, which records that it took them from
the pinned `gguf` package rather than from memory. The gaps are real: removed formats are not
reused, so an unknown number is reported as unknown rather than guessed at.

WHAT IS DELIBERATELY NOT READ

Array values are counted and not materialised. A tokeniser vocabulary is a 150,000-element array
of strings inside this header, and the one question this module answers does not need it. Reading
it would turn a metadata peek into hundreds of megabytes of Python strings.
"""
from __future__ import annotations

import struct
from pathlib import Path

#: The four bytes every GGUF starts with.
MAGIC = b"GGUF"

#: v1 used 32-bit lengths and nothing has produced one for years. Refusing it is honest: reading a
#: v1 file with v2 arithmetic would misparse silently, which is worse than declining.
SUPPORTED_VERSIONS = (2, 3)

#: A hard ceiling on the header read, so a corrupt length field cannot make us allocate a model's
#: worth of memory to answer a question about metadata.
MAX_HEADER_BYTES = 64 * 1024 * 1024

#: A single string or array length past this is a corrupt file rather than a big one. These numbers
#: come off disk and may be garbage, and `read(n)` on garbage is how a header read becomes an
#: out-of-memory kill.
MAX_ELEMENTS = 64 * 1024 * 1024

#: A sane ceiling on how many tensors a file may claim. A corrupt count would otherwise drive a
#: loop for as long as the header buffer lasts.
MAX_TENSORS = 1 << 20

# GGUF metadata value types, from the format specification.
(_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL, _STRING, _ARRAY,
 _UINT64, _INT64, _FLOAT64) = range(13)

_FIXED = {
    _UINT8: ("<B", 1), _INT8: ("<b", 1), _UINT16: ("<H", 2), _INT16: ("<h", 2),
    _UINT32: ("<I", 4), _INT32: ("<i", 4), _FLOAT32: ("<f", 4), _BOOL: ("<?", 1),
    _UINT64: ("<Q", 8), _INT64: ("<q", 8), _FLOAT64: ("<d", 8),
}

#: `general.file_type`: the WHOLE-FILE recipe label, from llama.cpp's LlamaFileType enum. This is
#: the number that says "this file calls itself Q4_K_M". It is not the per-tensor enum below.
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0", 1024: "GUESSED",
}

#: GGML tensor types: the PER-TENSOR enum. Different numbering from the table above, and mixing
#: them is the trap this module's docstring names. 8 is Q8_0 here and Q5_0 there.
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0",
}

#: Where a GGUF keeps its chat template. A file without one is handed raw text by every runtime
#: that reads this key, which is llama.cpp, Ollama and vLLM.
CHAT_TEMPLATE_KEY = "tokenizer.chat_template"

#: The metadata keys worth lifting. Deliberately a closed list: a header carries hundreds of keys
#: and the adapter answers a narrow question, so taking everything would be carrying a tokeniser
#: around to report a quantisation.
WANTED_KEYS = (
    "general.architecture", "general.file_type", "general.name",
    "general.quantization_version", "general.basename", "general.size_label",
    CHAT_TEMPLATE_KEY,
)

#: Filename spellings, longest first so `Q4_K_M` is not matched as `Q4_K` and then reported as
#: disagreeing with itself.
_NAME_CLAIMS = sorted((n for n in FILE_TYPES.values() if n != "GUESSED"), key=len, reverse=True)


class GGUFError(Exception):
    """Raised when a file is not a GGUF this reader can read, with an actionable reason."""


class _Cursor:
    """A bounded forward reader over the header bytes.

    Every read is length-checked against what was actually loaded, so a truncated file produces one
    sentence naming the shortfall rather than a `struct.error` from somewhere in the middle.
    """

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def take(self, n: int) -> bytes:
        if n < 0:
            raise GGUFError("the file claims a negative length, so it is corrupt")
        end = self.pos + n
        if end > len(self.buf):
            raise GGUFError(
                f"the file ends sooner than its own header says it should: {n} more bytes were "
                f"needed at offset {self.pos} and only {len(self.buf) - self.pos} are there. "
                f"That usually means a truncated or part-downloaded file.")
        out = self.buf[self.pos:end]
        self.pos = end
        return out

    def fixed(self, fmt: str, size: int):
        return struct.unpack(fmt, self.take(size))[0]

    def length(self) -> int:
        n = self.fixed("<Q", 8)
        if n > MAX_ELEMENTS:
            raise GGUFError(
                f"the file claims a single value of {n} elements, past the {MAX_ELEMENTS} this "
                f"tool will read. A length that large is a corrupt field rather than a big model.")
        return n

    def string(self) -> str:
        return self.take(self.length()).decode("utf-8", "replace")

    def value(self, vtype: int, *, materialise: bool):
        """One metadata value. `materialise=False` skips it and returns a description instead."""
        if vtype in _FIXED:
            fmt, size = _FIXED[vtype]
            return self.fixed(fmt, size)
        if vtype == _STRING:
            n = self.length()
            raw = self.take(n)
            return raw.decode("utf-8", "replace") if materialise else f"<string, {n} bytes>"
        if vtype == _ARRAY:
            inner = self.fixed("<I", 4)
            count = self.length()
            # COUNTED AND NOT MATERIALISED. See the module docstring: a tokeniser vocabulary lives
            # here, and the question this module answers does not need it.
            for _ in range(count):
                self.value(inner, materialise=False)
            return {"array_of": GGUF_VALUE_NAMES.get(inner, f"type {inner}"), "count": count}
        raise GGUFError(
            f"the file uses metadata value type {vtype}, which this reader does not know. "
            f"That is either a newer GGUF than this build understands or a corrupt field, and "
            f"guessing which would risk reporting a wrong answer confidently.")


#: Names for the value-type numbers, so an array's description reads as prose.
GGUF_VALUE_NAMES = {
    _UINT8: "uint8", _INT8: "int8", _UINT16: "uint16", _INT16: "int16", _UINT32: "uint32",
    _INT32: "int32", _FLOAT32: "float32", _BOOL: "bool", _STRING: "string", _ARRAY: "array",
    _UINT64: "uint64", _INT64: "int64", _FLOAT64: "float64",
}


def looks_like_gguf(path) -> bool:
    """True when the first four bytes are the GGUF magic.

    Cheap and total: used to decide which loader a path goes to, so it must not raise on a
    directory, an empty file or one that cannot be opened.
    """
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == MAGIC
    except OSError:
        return False


def claimed_quant_from_name(name: str):
    """The quantisation a filename claims, or None.

    Longest match first, so `Q4_K_M` is not read as `Q4_K`. Case-insensitive because publishers
    spell these both ways, and the separator before it is not required: real filenames carry
    `-Q4_K_M.gguf`, `.Q4_K_M.gguf` and `_q4_k_m.gguf`.
    """
    upper = str(name).upper()
    for claim in _NAME_CLAIMS:
        if claim in upper:
            return claim
    return None


def read_header(path, *, max_header_bytes: int = MAX_HEADER_BYTES) -> dict:
    """What the file says about itself: version, counts, selected metadata, a type census.

    Raises `GGUFError` with a readable reason for anything it cannot read, and never returns a
    partial answer: a header that parsed halfway is not evidence about a file.
    """
    p = Path(path)
    try:
        with open(p, "rb") as fh:
            buf = fh.read(max_header_bytes)
    except OSError as e:
        raise GGUFError(f"could not read the file: {e}") from e

    if len(buf) < 24:
        raise GGUFError(
            f"this file is {len(buf)} bytes, which is too short to be a GGUF: the header alone "
            f"needs 24 before any metadata.")
    cur = _Cursor(buf)
    if cur.take(4) != MAGIC:
        raise GGUFError(
            "this file does not start with the four bytes GGUF, so it is not a GGUF file. "
            "A safetensors checkpoint, a zip or an HTML error page saved by mistake all land here.")
    version = cur.fixed("<I", 4)
    if version not in SUPPORTED_VERSIONS:
        raise GGUFError(
            f"this is GGUF version {version} and this tool reads versions "
            f"{' and '.join(str(v) for v in SUPPORTED_VERSIONS)}. Version 1 used different length "
            f"fields, so reading it with these would misparse rather than fail, which is why it "
            f"is declined instead.")
    tensor_count = cur.fixed("<Q", 8)
    kv_count = cur.fixed("<Q", 8)
    if tensor_count > MAX_TENSORS:
        raise GGUFError(
            f"the file claims {tensor_count} tensors, past the {MAX_TENSORS} this tool will walk. "
            f"That is a corrupt count rather than a large model.")
    if kv_count > MAX_ELEMENTS:
        raise GGUFError(f"the file claims {kv_count} metadata entries, which is corrupt.")

    metadata = {}
    for _ in range(kv_count):
        key = cur.string()
        vtype = cur.fixed("<I", 4)
        wanted = key in WANTED_KEYS
        got = cur.value(vtype, materialise=wanted)
        if wanted:
            metadata[key] = got

    # The per-tensor census, which is the half a filename cannot tell you.
    census: dict[str, int] = {}
    unknown_types: dict[int, int] = {}
    for _ in range(tensor_count):
        cur.string()                                   # tensor name, not retained
        n_dims = cur.fixed("<I", 4)
        if n_dims > 8:
            raise GGUFError(
                f"a tensor claims {n_dims} dimensions, which no real model has, so the header is "
                f"corrupt or this reader has lost its place in it.")
        for _ in range(n_dims):
            cur.fixed("<Q", 8)
        ttype = cur.fixed("<I", 4)
        cur.fixed("<Q", 8)                             # data offset, not retained
        name = GGML_TYPES.get(ttype)
        if name is None:
            unknown_types[ttype] = unknown_types.get(ttype, 0) + 1
            name = f"unknown type {ttype}"
        census[name] = census.get(name, 0) + 1

    file_type_num = metadata.get("general.file_type")
    return {
        "version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
        "architecture": metadata.get("general.architecture"),
        "name": metadata.get("general.name"),
        "file_type_number": file_type_num if isinstance(file_type_num, int) else None,
        # None, not a guess, when the number is absent or outside the enum. A removed format's
        # number is not reused, so an unrecognised one means "newer than this table" and saying
        # so beats naming the nearest neighbour.
        "file_type": (FILE_TYPES.get(file_type_num)
                      if isinstance(file_type_num, int) else None),
        "quantization_version": metadata.get("general.quantization_version"),
        "has_chat_template": bool(metadata.get(CHAT_TEMPLATE_KEY)),
        "tensor_type_census": dict(sorted(census.items())),
        "unknown_tensor_types": dict(sorted(unknown_types.items())),
        "header_bytes_read": cur.pos,
        # NO `truncated` FLAG, and the first draft had one that was wrong in a way worth recording.
        # It reported `len(buf) == max_header_bytes`, which is true of every file bigger than the
        # cap and says nothing at all about the header: a 105 MB model whose header is 1.8 MB
        # reported "truncated" while having been read completely. A header that genuinely does not
        # fit cannot reach this line, because the cursor raises on the first read past the buffer.
        # So arriving here IS the evidence that the header fitted, and a field claiming otherwise
        # was a false alarm on every large file.
    }


__all__ = [
    "CHAT_TEMPLATE_KEY", "FILE_TYPES", "GGML_TYPES", "MAGIC", "MAX_HEADER_BYTES",
    "SUPPORTED_VERSIONS", "GGUFError", "claimed_quant_from_name", "looks_like_gguf",
    "read_header",
]
