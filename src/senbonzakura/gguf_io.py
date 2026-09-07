# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Read a GGUF file's header, and say what the file actually is.

WHY THIS IS HAND-ROLLED AND NOT THE `gguf` PACKAGE

Everything here reads the first few kilobytes of a file and stops. That makes it cheap enough to
run on every download, every conversion and every quantisation without anyone deciding it is too
slow to bother with, and a check nobody runs is not a check. Pulling a dependency in to do a
`struct.unpack` would also put a package between us and the one thing we most need to be certain
about, which is what a file on disk really is.

The `gguf` package is the right tool for WRITING one, and this module deliberately does not try.

WHAT IT IS FOR

A truncated download is caught by comparing bytes against the published length, and that is the
check that already exists. It does not catch the other two ways a file can be wrong:

  * a complete file that is not a GGUF at all (an HTML error page saved under a .gguf name is the
    classic, and it is exactly the size the server said it would be)
  * a complete, valid GGUF that is the WRONG VARIANT. A Q8_0 downloaded where a Q4_K_M was
    expected loads without complaint, serves without complaint, and silently changes every speed
    and memory number in a comparison.

The second is the one worth the code. `general.file_type` is a file-level label, and comparing it
against the quantisation named in the filename catches the mismatch in milliseconds.

WHY THE PASS/FAIL CHECK IS FILE-LEVEL AND NOT PER-TENSOR

A genuine Q4_K_M is a MIXTURE: most tensors are Q4_K and some are Q6_K, by design. Checking each
tensor's type against "Q4_K" would therefore fail every honest Q4_K_M ever produced. Failing a
correct file is worse than not checking, because it trains whoever meets it to pass --force.

That reasoning is about a VERDICT on a whole file, and it still stands. It says nothing against
reading the per-tensor types, and `read_tensor_info` does, for the one question the file-level
label cannot answer: when somebody asks for a specific tensor to be kept at a specific precision,
did that actually happen? `general.file_type` is unchanged by such an override, so the only
evidence is the tensor itself. Asking for something and checking you got it is a different
activity from judging a file against its name, and only the second can fail a correct file.
"""
from __future__ import annotations

import struct
from pathlib import Path

#: The four bytes every GGUF starts with.
MAGIC = b"GGUF"

#: Versions this reader understands. v1 used 32-bit lengths and no v1 file has been produced for
#: years; refusing it is honest, because reading it with v2 arithmetic would silently misparse.
SUPPORTED_VERSIONS = (2, 3)

#: How much of the header to read. The metadata block on a large model runs to a few hundred
#: kilobytes (a 150k-entry tokeniser vocabulary lives in here), and this is a hard ceiling so a
#: corrupt length field cannot make us allocate a model's worth of RAM to read its header.
MAX_HEADER_BYTES = 64 * 1024 * 1024

#: A single string or array length past this is a corrupt file, not a big one. Bounded because
#: these numbers come off disk and are attacker-controlled in the sense that matters: they may be
#: garbage, and `read(n)` on garbage is how a header read becomes an out-of-memory kill.
MAX_ELEMENTS = 64 * 1024 * 1024

# GGUF metadata value types, from the format specification.
(_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL, _STRING, _ARRAY,
 _UINT64, _INT64, _FLOAT64) = range(13)

_FIXED = {
    _UINT8: ("<B", 1), _INT8: ("<b", 1), _UINT16: ("<H", 2), _INT16: ("<h", 2),
    _UINT32: ("<I", 4), _INT32: ("<i", 4), _FLOAT32: ("<f", 4), _BOOL: ("<?", 1),
    _UINT64: ("<Q", 8), _INT64: ("<q", 8), _FLOAT64: ("<d", 8),
}

#: `general.file_type`, verbatim from llama.cpp's LlamaFileType enum (checked against gguf 0.19.0
#: rather than transcribed from memory). The gaps are real: 4 to 6 were removed formats and are
#: not reused, so an unknown number must be reported as unknown rather than guessed at.
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0", 1024: "GUESSED",
}

#: GGML tensor types, the per-TENSOR enum. Not the same numbering as `general.file_type` above and
#: the two must never be looked up in each other's table: 8 is Q8_0 here and Q5_0 there. Read out
#: of the pinned `gguf` package's GGMLQuantizationType rather than typed from memory, and the gaps
#: (3 to 5, 31 to 33, 36 to 38) are removed formats that are not reused, so an unknown number is
#: reported as unknown rather than guessed at.
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0",
}

#: The types `llama-quantize` will accept for a per-tensor override. Its own list is wider than
#: what is useful here; these are the ones that make sense to pin a single tensor to, and naming a
#: closed set means a typo is refused before a multi-hour job starts rather than after it.
OVERRIDE_TYPES = tuple(sorted(
    {"F32", "F16", "BF16", "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K",
     "Q5_1", "Q5_0", "Q4_1", "Q4_0", "IQ4_NL", "IQ4_XS"}))

#: Filename spellings to the file type they claim, longest first so "Q4_K_M" is not matched as
#: "Q4_K" and then reported as a mismatch against itself.
_NAME_CLAIMS = sorted(
    {name: name for name in FILE_TYPES.values() if name != "GUESSED"}.items(),
    key=lambda kv: -len(kv[0]))


class GGUFError(Exception):
    """A file that is not the GGUF it was supposed to be, phrased for a person."""


class _Cursor:
    """A bounds-checked reader over the header bytes.

    Every read is checked against the buffer end rather than relying on slicing, which returns a
    short result silently and turns a truncated header into a plausible-looking parse.
    """

    def __init__(self, buf, *, capped=False):
        self.buf, self.pos = buf, 0
        # Whether the buffer ends because the FILE ended or because our read window did. Running
        # out is the same event in both cases and means opposite things, and the wrong one of the
        # two accuses a healthy file of being corrupt. A real 1.2B model's header is over a
        # megabyte on its own: 65,536 tokeniser entries and their merges live in the metadata,
        # which is far larger than the format's fixed fields would lead anyone to expect.
        self.capped = capped

    def take(self, n):
        if n < 0 or self.pos + n > len(self.buf):
            if self.capped:
                raise GGUFError(
                    f"the header did not fit in the {len(self.buf):,} bytes read from the front "
                    f"of the file, so it could not be parsed. This is a limit on the read, NOT "
                    f"evidence that the file is damaged: a large tokeniser vocabulary lives in "
                    f"the metadata and pushes the header past small windows. Raise max_bytes.")
            raise GGUFError(
                f"the header ends after {len(self.buf)} bytes but the file's own length fields "
                f"ask for {self.pos + max(0, n)}. The file is truncated or is not a GGUF.")
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def scalar(self, fmt, size):
        return struct.unpack(fmt, self.take(size))[0]

    def count(self):
        n = self.scalar("<Q", 8)
        if n > MAX_ELEMENTS:
            raise GGUFError(
                f"a length field claims {n:,} elements, past the {MAX_ELEMENTS:,} cap this "
                f"reader will honour. A header does not legitimately contain that, so the file "
                f"is corrupt rather than large.")
        return n

    def string(self):
        return self.take(self.count()).decode("utf-8", errors="replace")

    def value(self, vtype):
        if vtype in _FIXED:
            return self.scalar(*_FIXED[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            etype = self.scalar("<I", 4)
            n = self.count()
            if etype == _ARRAY:
                raise GGUFError("nested arrays are not part of the GGUF format")
            # Read them rather than skip: the element widths differ and a wrong skip would
            # desynchronise every key after this one while still parsing.
            return [self.value(etype) for _ in range(n)]
        raise GGUFError(f"unknown GGUF metadata value type {vtype}")


def _read_head(path, max_bytes):
    """The parse itself, returning the header AND the cursor sitting on the tensor info section.

    Split out so `read_tensor_info` continues where `read_header` stopped rather than keeping a
    second copy of the metadata walk. Two parsers over one format is how a reader ends up
    disagreeing with itself about the same file.
    """
    p = Path(path)
    try:
        with open(p, "rb") as f:
            head = f.read(max_bytes)
    except OSError as e:
        raise GGUFError(f"could not read {p}: {e}") from e

    if len(head) < 24:
        raise GGUFError(
            f"{p} is {len(head)} bytes, too short to hold even a GGUF header. A zero-length or "
            f"tiny file here is usually a failed download that left the name behind.")
    if head[:4] != MAGIC:
        # The single most useful diagnostic, because the commonest cause is an error page.
        looks_like = "an HTML page" if head.lstrip()[:1] == b"<" else "not a GGUF"
        raise GGUFError(
            f"{p} does not start with the GGUF magic bytes; it is {looks_like}. A download that "
            f"returned an error page saves at exactly the length the server promised, so a "
            f"size check alone will not catch this.")

    # `capped` when the read stopped because the window filled rather than because the file
    # ended, which is the difference between "your file is broken" and "read more of it".
    c = _Cursor(head, capped=len(head) >= max_bytes)
    c.take(4)
    version = c.scalar("<I", 4)
    if version not in SUPPORTED_VERSIONS:
        raise GGUFError(
            f"{p} is GGUF version {version}, and this reader understands "
            f"{' and '.join(map(str, SUPPORTED_VERSIONS))}. Parsing it with the wrong version's "
            f"arithmetic would misread every field rather than fail, so it stops here.")
    tensor_count = c.count()
    kv_count = c.count()

    metadata = {}
    for _ in range(kv_count):
        key = c.string()
        metadata[key] = c.value(c.scalar("<I", 4))

    ftype_id = metadata.get("general.file_type")
    info = {
        "path": str(p),
        "version": version,
        "tensor_count": tensor_count,
        "metadata": metadata,
        "architecture": metadata.get("general.architecture"),
        "file_type_id": ftype_id,
        "file_type": FILE_TYPES.get(ftype_id) if isinstance(ftype_id, int) else None,
    }
    return info, c


def read_header(path, *, max_bytes=MAX_HEADER_BYTES):
    """Metadata and tensor count from a GGUF, reading only its head.

    Returns {"version", "tensor_count", "metadata", "architecture", "file_type",
    "file_type_id"}. Raises GGUFError with a readable reason for anything that is not a GGUF this
    reader can parse.
    """
    return _read_head(path, max_bytes)[0]


#: A tensor name past this is corrupt rather than long. llama.cpp's own limit is 64 and this is
#: generous against it, because refusing a legitimate file is the worse failure of the two.
MAX_TENSOR_NAME = 512

#: More dimensions than any tensor in a language model has. GGUF allows four; this refuses a
#: length field that would otherwise make the reader walk off into the data section and report
#: nonsense with a straight face.
MAX_TENSOR_DIMS = 4


def read_tensor_info(path, *, max_bytes=MAX_HEADER_BYTES):
    """Every tensor's name, shape and TYPE, without reading a byte of tensor data.

    Returns a list of {"name", "dims", "type_id", "type", "offset"} in file order. `type` is None
    for a type id this reader does not know, which is reported rather than guessed: naming an
    unknown number would make a wrong answer look like an answer.

    `dims` is in GGUF's own order, fastest-moving axis first, which is the REVERSE of the shape
    the same tensor has in numpy or torch. Reported as the file stores it rather than flipped to
    suit a caller, because this module's job is to say what is on disk.

    The tensor info section sits immediately after the metadata, so this costs the same head read
    `read_header` already does. It exists to answer "did the precision I asked for actually land
    on the tensor I named", which the file-level label cannot: an override leaves
    `general.file_type` completely unchanged.
    """
    _info, c = _read_head(path, max_bytes)
    out = []
    for i in range(_info["tensor_count"]):
        n = c.count()
        if n > MAX_TENSOR_NAME:
            raise GGUFError(
                f"tensor {i} claims a name {n:,} bytes long, past the {MAX_TENSOR_NAME} this "
                f"reader will honour. The tensor info section is misaligned or the file is "
                f"corrupt; either way what follows would be read as garbage.")
        name = c.take(n).decode("utf-8", errors="replace")
        n_dims = c.scalar("<I", 4)
        if n_dims > MAX_TENSOR_DIMS:
            raise GGUFError(
                f"tensor {name!r} claims {n_dims} dimensions and GGUF allows at most "
                f"{MAX_TENSOR_DIMS}. Reading on would walk into the data section.")
        dims = [c.scalar("<Q", 8) for _ in range(n_dims)]
        type_id = c.scalar("<I", 4)
        offset = c.scalar("<Q", 8)
        out.append({
            "name": name,
            "dims": dims,
            "type_id": type_id,
            "type": GGML_TYPES.get(type_id),
            "offset": offset,
        })
    return out


def type_census(path, *, max_bytes=MAX_HEADER_BYTES):
    """How many tensors carry each type, highest count first.

    The honest way to describe what a quantisation actually produced. "Q4_K_M" is a recipe name;
    this is what came out of it, and the two are not the same statement.
    """
    counts = {}
    for t in read_tensor_info(path, max_bytes=max_bytes):
        key = t["type"] or f"unknown type {t['type_id']}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def tensor_types(path, names, *, max_bytes=MAX_HEADER_BYTES):
    """The type of each named tensor, or None where the file has no such tensor.

    None means ABSENT and never "some default". A tied-embedding model has no `output.weight` at
    all, so an override aimed at it silently does nothing, and a caller that could not tell absent
    from present would report that silence as success.
    """
    have = {t["name"]: t["type"] for t in read_tensor_info(path, max_bytes=max_bytes)}
    return {n: have.get(n) for n in names}


def claimed_quant(name):
    """The quantisation a filename claims, or None if it does not claim one.

    Matched longest-first and case-insensitively against the known file types, so
    `model-Q4_K_M.gguf` claims Q4_K_M rather than Q4_K. A filename that names nothing is not an
    error: plenty of legitimate files are simply called `model.gguf`.
    """
    stem = Path(name).name.upper()
    for claim, _ in _NAME_CLAIMS:
        if claim in stem:
            return claim
    return None


def verify(path, *, expect_quant=None, expect_arch=None, min_tensors=1):
    """Read the header and check the file is what it is supposed to be. Returns the header.

    `expect_quant` of None means "whatever the filename claims", which is the useful default: it
    turns every download of a conventionally-named file into a checked one for free. Pass a value
    to assert it explicitly, or `False` to skip the quantisation check entirely.
    """
    head = read_header(path)

    if head["tensor_count"] < min_tensors:
        raise GGUFError(
            f"{path} holds {head['tensor_count']} tensors, fewer than the {min_tensors} expected. "
            f"A container with a valid header and no weights in it loads and answers nonsense.")

    if expect_arch and head["architecture"] != expect_arch:
        raise GGUFError(
            f"{path} reports architecture {head['architecture']!r}, not the expected "
            f"{expect_arch!r}. This is a different model from the one asked for.")

    if expect_quant is False:
        return head
    want = expect_quant if expect_quant is not None else claimed_quant(path)
    if want is None:
        return head
    got = head["file_type"]
    if got is None:
        raise GGUFError(
            f"{path} claims {want} in its name but records no readable general.file_type "
            f"(raw value {head['file_type_id']!r}), so the claim cannot be checked. Treat the "
            f"file as unidentified rather than as the quantisation its name asserts.")
    if got != want:
        raise GGUFError(
            f"{path} is a {got} file but its name says {want}. It is complete and it will load, "
            f"which is what makes this worth checking: the wrong quantisation changes every "
            f"speed and memory figure it appears in, and nothing downstream would report it.")
    return head
