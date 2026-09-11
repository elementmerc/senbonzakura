# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The last of shortlist item A: downloads, manifests, and byte comparison.

Same argument as the two files before it, and this batch is mostly about what happens when input
this process did not write turns out to be wrong. `read_manifest` is the sharpest example: it
reads a file on disk that decides which rows every published number is measured on, so anything
that is not the shape we understand has to be refused rather than interpreted loosely.

The download paths matter for a different reason. Every one of them is about not leaving a
partial file behind, because a `.part` that the next run mistakes for a finished download is a
corrupted input that looks like a cached one.

CPU only, no network: every transfer here is driven through a fake `urlopen`.
"""
import io
import json
import urllib.error
import urllib.request

import pytest

from senbonzakura import convert, fetch, resources, track

# ── resources.cuda_free_total: the card-wide reading the governor reacts to ──────────────────


class _Torch:
    def __init__(self, available=True, free=1, total=2, boom=False):
        class _C:
            @staticmethod
            def is_available():
                return available

            @staticmethod
            def mem_get_info(_i):
                if boom:
                    raise RuntimeError("driver gone")
                return (free, total)

        self.cuda = _C


def test_free_total_reads_the_whole_card_not_our_share(monkeypatch):
    """`mem_get_info` reports the WHOLE card, so it sees VRAM a game has taken.

    That is the point rather than an accident: the governor has to react to memory this process
    does not own, which on WSL2 is the only way a Windows-side game is visible at all.
    """
    monkeypatch.setattr(resources, "_torch", lambda: _Torch(free=2_000, total=6_000))
    assert resources.cuda_free_total("cuda:0") == (2_000, 6_000)


@pytest.mark.parametrize(("device", "torch_kwargs", "why"), [
    ("cpu", {}, "not a cuda device"),
    (object(), {}, "not even a string"),
    ("cuda:0", {"available": False}, "torch is present but there is no card"),
    ("cuda:0", {"boom": True}, "the driver refused the query"),
])
def test_free_total_is_none_when_the_question_cannot_be_answered(device, torch_kwargs, why,
                                                                 monkeypatch):
    """None rather than a guess. A governor handed a made-up number acts on it."""
    monkeypatch.setattr(resources, "_torch", lambda: _Torch(**torch_kwargs))
    assert resources.cuda_free_total(device) is None, why


# ── fetch.download: the scheme guard, and never leaving a partial ────────────────────────────

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def test_a_plain_http_url_is_refused_at_the_point_of_use(tmp_path):
    """RE-CHECKED HERE, not only where it was parsed.

    `parse_source` is the only caller today, and a second one would not inherit its guarantee.
    Two checks satisfying the same rule is the right shape; one check and a comment is not.
    """
    with pytest.raises(fetch.FetchError, match="https is required"):
        fetch.download("url", "http://example.com/f.gguf", "f.gguf", tmp_path,
                       log=lambda *_a: None)


def test_loopback_over_plain_http_is_allowed(tmp_path, monkeypatch):
    """The carve-out, and the reason for it: there is no network path to intercept."""
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _Resp(b"payload"))
    got = fetch.download("url", "http://127.0.0.1:8000/f.bin", "f.bin", tmp_path,
                         log=lambda *_a: None)
    assert got.read_bytes() == b"payload"


def test_a_finished_download_leaves_no_part_file(tmp_path, monkeypatch):
    """The `.part` is renamed into place, so a completed transfer leaves exactly one file."""
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _Resp(b"x" * (fetch.CHUNK + 17)))
    got = fetch.download("url", "https://example.com/f.bin", "f.bin", tmp_path,
                         log=lambda *_a: None)
    assert got.stat().st_size == fetch.CHUNK + 17
    assert not (tmp_path / "f.bin.part").exists()


def test_a_failed_download_removes_the_partial_and_explains(tmp_path, monkeypatch):
    """A `.part` the next run mistakes for a finished file is a corrupted input that looks cached."""
    def _boom(*_a, **_k):
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    with pytest.raises(fetch.FetchError, match="could not download"):
        fetch.download("url", "https://example.com/f.bin", "f.bin", tmp_path,
                       log=lambda *_a: None)
    assert not (tmp_path / "f.bin.part").exists()
    assert not (tmp_path / "f.bin").exists()


def test_an_http_error_is_closed_rather_than_leaking_its_socket(tmp_path, monkeypatch):
    """THE SUITE RUNS WITH -W error::ResourceWarning BECAUSE OF THIS ONE.

    `HTTPError` is a file-like response object as well as an exception, so it holds a socket until
    closed. The `e.close()` in the handler is why this test can exist without turning the run red.
    """
    closed = []

    class _HTTPError(urllib.error.HTTPError):
        def close(self):
            closed.append(True)
            # AND ACTUALLY CLOSE IT. Recording the call without closing leaves the temporary file
            # HTTPError wraps to be reaped at collection, which raises a ResourceWarning from the
            # garbage collector. The first version of this test did exactly that and leaked the
            # very thing it was written to prove does not leak.
            super().close()

    def _boom(*_a, **_k):
        raise _HTTPError("https://example.com/f.bin", 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    with pytest.raises(fetch.FetchError):
        fetch.download("url", "https://example.com/f.bin", "f.bin", tmp_path,
                       log=lambda *_a: None)
    assert closed, "the error response was never closed, so its socket leaked"


def test_an_interrupt_mid_download_still_clears_the_partial(tmp_path, monkeypatch):
    """The bare `except BaseException` branch: Ctrl+C must not leave a half file behind.

    Baseline section 2.1 asks for exactly this and it had never been exercised. It re-raises, so
    the interrupt is not swallowed; only the file is cleaned up.
    """
    def _boom(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    with pytest.raises(KeyboardInterrupt):
        fetch.download("url", "https://example.com/f.bin", "f.bin", tmp_path,
                       log=lambda *_a: None)
    assert not (tmp_path / "f.bin.part").exists()


# ── track.read_manifest: untrusted input that decides which rows get measured ────────────────

def test_no_manifest_at_all_is_a_legal_state(tmp_path):
    """A hand-built track predates the builder and still has to run. None means "unknown"."""
    assert track.read_manifest(tmp_path) is None


def test_an_unparseable_manifest_reads_as_no_manifest(tmp_path):
    (tmp_path / "track.json").write_text("{not json", encoding="utf-8")
    assert track.read_manifest(tmp_path) is None


@pytest.mark.parametrize(("doc", "why"), [
    ([1, 2, 3], "a list where an object was expected"),
    ({"counts": None}, "null counts, which reached a .get on None deep in the boundary check"),
    ({"counts": [1, 2]}, "counts that are not a mapping"),
    ({}, "no counts at all"),
])
def test_a_manifest_of_the_wrong_shape_is_no_manifest_rather_than_one_to_interpret(doc, why,
                                                                                   tmp_path):
    """`"counts" in m` alone let a null through. The shape is checked, not the key's presence."""
    (tmp_path / "track.json").write_text(json.dumps(doc), encoding="utf-8")
    assert track.read_manifest(tmp_path) is None, why


def test_a_known_schema_is_returned(tmp_path):
    doc = {"counts": {"bad_ds": 1}, "schema": min(track.KNOWN_SCHEMAS)}
    (tmp_path / "track.json").write_text(json.dumps(doc), encoding="utf-8")
    assert track.read_manifest(tmp_path) == doc


def test_a_manifest_with_no_schema_field_gets_the_original_one(tmp_path):
    """Written before the field existed. Defaulting is right; guessing at an unknown one is not."""
    (tmp_path / "track.json").write_text(json.dumps({"counts": {"bad_ds": 1}}), encoding="utf-8")
    assert track.read_manifest(tmp_path) is not None


@pytest.mark.parametrize("schema", ["senbonzakura-track/99", ["a", "list"], 7])
def test_an_unknown_schema_stops_the_run_rather_than_slicing_at_a_guess(schema, tmp_path):
    """THE REASON THIS FIELD IS READ AT ALL.

    Until 2026-08-03 it was written and never read, which is worse than not having one: it looks
    like a compatibility guarantee and is not. An older build would take the fields it recognised,
    ignore whatever changed, and slice the datasets confidently at the wrong offsets, and every
    number downstream would come from the wrong rows with nothing saying so.

    A list-valued schema is in the parameters because it used to reach a frozenset membership test
    and raise an unhashable-type TypeError instead of this refusal.
    """
    (tmp_path / "track.json").write_text(
        json.dumps({"counts": {"bad_ds": 1}, "schema": schema}), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not understand"):
        track.read_manifest(tmp_path)


# ── track.read_prompts: a typo in a filename must not get a paragraph about Hub ids ──────────

def test_a_missing_file_that_is_plainly_a_file_keeps_the_direct_error(tmp_path):
    """Routing it to the resolver would answer a typo with an essay about dataset ids."""
    with pytest.raises(SystemExit, match="no such file"):
        track.read_prompts(tmp_path / "harmfull.txt")


def test_a_plain_file_is_read_verbatim_line_by_line(tmp_path):
    """The documented format, and a corpus must not change under a user who did nothing."""
    p = tmp_path / "prompts.txt"
    p.write_text("  first  \nsecond\n\nthird\n", encoding="utf-8")
    assert track.read_prompts(p) == ["first", "second", "", "third"]


def test_an_unreadable_prompt_file_says_so_plainly(tmp_path, monkeypatch):
    """The OSError branch, which had never been taken."""
    p = tmp_path / "prompts.txt"
    p.write_text("x", encoding="utf-8")

    def _boom(*_a, **_k):
        raise OSError("permission denied")

    monkeypatch.setattr(type(p), "read_text", _boom)
    with pytest.raises(SystemExit, match="could not read"):
        track.read_prompts(p)


def test_a_resolver_failure_is_reported_without_a_traceback(tmp_path, monkeypatch):
    """Anything that is not a plain file goes through the shared resolver, and its errors are
    the user's problem to act on rather than a stack to read.
    """
    monkeypatch.setattr(track.dataset, "resolve",
                        lambda *a, **k: (_ for _ in ()).throw(
                            track.dataset.DatasetError("no such dataset")))
    with pytest.raises(SystemExit, match="no such dataset"):
        track.read_prompts("someone/a-dataset")


# ── convert.raw_bytes_equal: byte equality, stated exactly ───────────────────────────────────

def _shard(path, header, payload):
    """A minimal safetensors-shaped file: 8-byte header length, header, then the data section."""
    blob = json.dumps(header).encode()
    path.write_bytes(len(blob).to_bytes(8, "little") + blob + payload)


def test_tensors_of_different_lengths_differ_at_zero(tmp_path):
    """Short circuit before either file is opened: nothing to compare byte for byte."""
    a, b = tmp_path / "a.st", tmp_path / "b.st"
    _shard(a, {"t": {"data_offsets": [0, 4]}}, b"abcd")
    _shard(b, {"t": {"data_offsets": [0, 8]}}, b"abcdefgh")
    assert convert.raw_bytes_equal(a, {"data_offsets": [0, 4]},
                                   b, {"data_offsets": [0, 8]}) == (False, 0)


def test_identical_bytes_compare_equal(tmp_path):
    a, b = tmp_path / "a.st", tmp_path / "b.st"
    _shard(a, {"t": {"data_offsets": [0, 6]}}, b"abcdef")
    _shard(b, {"t": {"data_offsets": [0, 6]}}, b"abcdef")
    assert convert.raw_bytes_equal(a, {"data_offsets": [0, 6]},
                                   b, {"data_offsets": [0, 6]}) == (True, None)


def test_the_first_differing_offset_is_reported_not_just_the_fact(tmp_path):
    """The offset is the diagnostic. "They differ" sends somebody looking through a whole shard."""
    a, b = tmp_path / "a.st", tmp_path / "b.st"
    _shard(a, {"t": {"data_offsets": [0, 6]}}, b"abcdef")
    _shard(b, {"t": {"data_offsets": [0, 6]}}, b"abcXef")
    assert convert.raw_bytes_equal(a, {"data_offsets": [0, 6]},
                                   b, {"data_offsets": [0, 6]}) == (False, 3)


def test_a_difference_past_the_first_chunk_is_still_found(tmp_path):
    """The loop runs more than once. With a single chunk the `remaining`/`seen` arithmetic is
    never exercised, and an off-by-one there would report the wrong offset rather than no offset.
    """
    payload_a = b"a" * 10
    payload_b = b"a" * 7 + b"Z" + b"a" * 2
    a, b = tmp_path / "a.st", tmp_path / "b.st"
    _shard(a, {"t": {"data_offsets": [0, 10]}}, payload_a)
    _shard(b, {"t": {"data_offsets": [0, 10]}}, payload_b)
    assert convert.raw_bytes_equal(a, {"data_offsets": [0, 10]},
                                   b, {"data_offsets": [0, 10]}, chunk=4) == (False, 7)


def test_a_shard_that_ends_early_is_refused_rather_than_called_equal(tmp_path):
    """A truncated file must not read as "the tensors match" because both reads came back short.

    That is the failure shape this project keeps finding: a check that passes because nothing
    happened rather than because everything did.
    """
    a, b = tmp_path / "a.st", tmp_path / "b.st"
    _shard(a, {"t": {"data_offsets": [0, 16]}}, b"abcd")        # header promises 16, holds 4
    _shard(b, {"t": {"data_offsets": [0, 16]}}, b"abcd")
    with pytest.raises(convert.ConvertError, match="ended before its header said"):
        convert.raw_bytes_equal(a, {"data_offsets": [0, 16]}, b, {"data_offsets": [0, 16]})
