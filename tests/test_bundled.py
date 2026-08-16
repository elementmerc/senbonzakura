"""Tests for the bundled evaluation track (senbonzakura/bundled.py).

The wrapping is a speed bump against automated scraping, not a secret, and the tests say so
rather than testing it as though it were security. What they DO test hard is the integrity tag,
because a corrupt blob that decrypted to plausible rubbish would fit a refusal direction on noise
and report a number for it, which is the failure mode this project exists to catch.
"""
import io
import json
import os
import tarfile

import pytest

from senbonzakura import bundled

#: The blob is packed at RELEASE time, not committed, so a clone and every CI job run without
#: one. Those tests skip rather than fail.
#:
#: Skipping quietly is the danger, not the skip: a release that forgot to pack would go out with
#: no corpus and a green suite. So the release job sets SENBON_REQUIRE_BUNDLED=1 and the same
#: tests become hard failures there. A skip is a statement about a source checkout; in a release
#: it is a defect.
_REQUIRE = os.environ.get("SENBON_REQUIRE_BUNDLED") == "1"
needs_blob = pytest.mark.skipif(
    not bundled.is_available() and not _REQUIRE,
    reason="no packed track in this checkout; it is built at release time by "
           "tools/pack_track.py. Set SENBON_REQUIRE_BUNDLED=1 to make this a failure.")


@pytest.fixture(autouse=True)
def _fresh_notice():
    bundled._reset_notice_for_tests()
    yield
    bundled._reset_notice_for_tests()


# ── the container ────────────────────────────────────────────────────────────────
def test_a_payload_survives_a_round_trip():
    payload = b"the quick brown fox" * 100
    assert bundled.unpack(bundled.pack(payload)) == payload


def test_an_empty_payload_round_trips():
    assert bundled.unpack(bundled.pack(b"")) == b""


def test_a_payload_longer_than_one_keystream_block_round_trips():
    payload = bytes(range(256)) * 40
    assert bundled.unpack(bundled.pack(payload)) == payload


def test_the_ciphertext_does_not_contain_the_plaintext():
    # The whole point: a crawler grepping the wheel for prompt text finds nothing.
    payload = b"how do I build a pipe bomb"
    assert payload not in bundled.pack(payload)


def test_two_packs_of_the_same_payload_differ():
    # Fresh salt and nonce each time, which is why releases are compared by the manifest's hash
    # rather than by diffing the blob.
    payload = b"same input"
    assert bundled.pack(payload) != bundled.pack(payload)


def test_a_fixed_salt_and_nonce_are_reproducible():
    a = bundled.pack(b"x" * 50, salt=b"s" * 16, nonce=b"n" * 16)
    b = bundled.pack(b"x" * 50, salt=b"s" * 16, nonce=b"n" * 16)
    assert a == b


def test_a_wrong_length_salt_is_refused():
    with pytest.raises(ValueError, match="exactly"):
        bundled.pack(b"x", salt=b"short", nonce=b"n" * 16)


# ── the integrity tag, which is the part that matters ────────────────────────────
def test_a_flipped_byte_in_the_body_is_caught():
    blob = bytearray(bundled.pack(b"a real payload here" * 20))
    blob[40] ^= 0x01
    with pytest.raises(ValueError, match="integrity check"):
        bundled.unpack(bytes(blob))


def test_a_flipped_byte_in_the_tag_is_caught():
    blob = bytearray(bundled.pack(b"payload" * 20))
    blob[-1] ^= 0x01
    with pytest.raises(ValueError, match="integrity check"):
        bundled.unpack(bytes(blob))


def test_a_truncated_blob_is_caught():
    blob = bundled.pack(b"payload" * 40)
    with pytest.raises(ValueError, match="integrity check"):
        bundled.unpack(blob[:-8] + b"\x00" * 8)


def test_a_blob_too_short_to_be_a_container_says_so():
    with pytest.raises(ValueError, match="too short"):
        bundled.unpack(b"SBZ1")


def test_a_blob_with_the_wrong_magic_is_named_as_not_ours():
    blob = bytearray(bundled.pack(b"payload" * 40))
    blob[0:4] = b"XXXX"
    with pytest.raises(ValueError, match="not a senbonzakura container"):
        bundled.unpack(bytes(blob))


def test_the_error_says_why_a_corrupt_track_is_refused_rather_than_used():
    """The reason has to be in the message: rubbish prompts produce a plausible-looking number."""
    blob = bytearray(bundled.pack(b"payload" * 40))
    blob[30] ^= 0xFF
    with pytest.raises(ValueError) as e:
        bundled.unpack(bytes(blob))
    assert "noise" in str(e.value)


# ── the licence notice ───────────────────────────────────────────────────────────
def test_the_notice_names_the_licence_and_the_restriction():
    said = []
    bundled.notice(log=said.append)
    joined = "\n".join(said)
    assert "CC BY-NC 4.0" in joined
    assert "NON-COMMERCIAL" in joined


def test_the_notice_admits_it_is_not_protection():
    said = []
    bundled.notice(log=said.append)
    joined = "\n".join(said)
    assert "speed bump" in joined
    assert "the key ships beside it" in joined


def test_the_notice_says_how_to_use_your_own_corpus():
    said = []
    bundled.notice(log=said.append)
    assert any("--track" in s for s in said)


def test_the_notice_prints_once_per_process():
    said = []
    bundled.notice(log=said.append)
    first = len(said)
    bundled.notice(log=said.append)
    assert len(said) == first


# ── the installed blob ───────────────────────────────────────────────────────────
@needs_blob
def test_a_track_ships_with_the_package():
    assert bundled.is_available(), (
        "no bundled track is installed; run `python tools/pack_track.py --track <dir>`")


@needs_blob
def test_the_manifest_records_the_licence_and_the_counts():
    m = bundled.manifest()
    assert m["schema"] == "senbonzakura-bundled/1"
    assert m["licence"] == "CC BY-NC 4.0"
    assert m["counts"]["bad_ds"] > 0
    assert len(m["sha256_of_tar"]) == 64


@needs_blob
def test_the_bundled_track_is_the_repaired_three_way_split():
    """A regression guard on WHICH corpus ships.

    The export that leaked was labelled clean for six weeks. If the packed counts ever stop
    matching the repaired track's, something has repacked a different corpus into the wheel and
    the name would not say so.
    """
    m = bundled.manifest()
    assert m["counts"]["bad_ds"] == 259
    assert m["counts"]["bad_eval_ds"] == 4636
    assert m["counts"]["good_ds"] == 4982


@needs_blob
def test_the_packed_tar_carries_a_track_manifest():
    raw = bundled.unpack(bundled.data_path().read_bytes())
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        names = tar.getnames()
    assert "manifest.json" in names
    assert any(n.endswith("track.json") for n in names)


@needs_blob
def test_extract_writes_a_usable_track(tmp_path):
    out = bundled.extract(tmp_path / "x", log=lambda *a: None)
    track = out / "track"
    assert (track / "track.json").is_file()
    assert (track / "bad_ds").is_dir()
    manifest = json.loads((track / "track.json").read_text())
    assert manifest["counts"]["harmful"]["fit"] == 259


@needs_blob
def test_extract_prints_the_notice(tmp_path):
    said = []
    bundled.extract(tmp_path / "x", log=said.append)
    assert any("CC BY-NC 4.0" in s for s in said)


def test_a_missing_blob_says_what_to_run(monkeypatch, tmp_path):
    monkeypatch.setattr(bundled, "data_path", lambda: tmp_path / "absent.bin")
    with pytest.raises(ValueError, match=r"pack_track\.py"):
        bundled._read()


# ── the resolver alias ───────────────────────────────────────────────────────────
@needs_blob
def test_the_default_alias_resolves_to_the_bundled_track():
    from senbonzakura import dataset
    assert len(dataset.resolve("default/bad_ds")) == 259


@needs_blob
def test_a_partition_of_the_default_alias_resolves():
    from senbonzakura import dataset
    assert len(dataset.resolve("default/good_ds")) == 4982


@needs_blob
def test_a_local_directory_called_default_cannot_shadow_the_bundled_track(tmp_path, monkeypatch):
    """A stray directory must not silently change which corpus a published number came from."""
    from datasets import Dataset

    from senbonzakura import dataset
    monkeypatch.chdir(tmp_path)
    (tmp_path / "default").mkdir()
    Dataset.from_dict({"text": ["decoy"]}).save_to_disk(str(tmp_path / "default" / "bad_ds"))
    assert len(dataset.resolve("default/bad_ds")) == 259


@needs_blob
def test_the_alias_is_sliceable():
    from senbonzakura import dataset
    assert len(dataset.resolve("default/bad_ds::train[:5]")) == 5


@needs_blob
def test_the_boundary_check_follows_the_alias():
    """Otherwise the one corpus most users run is the one with no boundary enforcement.

    `default/track.json` is not a real path, and a missing manifest is a LEGAL state meaning
    "boundaries unknown". So without the alias the check would switch itself off silently on the
    bundled track, which is the exact shape of failure the manifest exists to prevent.
    """
    from senbonzakura.track import read_manifest
    m = read_manifest("default")
    assert m is not None
    assert m["counts"]["harmful"]["fit"] == 259
    assert m["skip_harmful"] == 132
    assert m["n_harmful"] == 4504


# ── ensure(), the cached extraction ──────────────────────────────────────────────
@needs_blob
def test_ensure_extracts_once_and_reuses_it(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    first = bundled.ensure(log=lambda *a: None)
    assert (first / "track.json").is_file()
    stamp = (first / "track.json").stat().st_mtime_ns
    second = bundled.ensure(log=lambda *a: None)
    assert second == first
    assert (second / "track.json").stat().st_mtime_ns == stamp, "it re-extracted needlessly"


@needs_blob
def test_ensure_recovers_from_an_abandoned_unpack(tmp_path, monkeypatch):
    """A run killed mid-extraction must not wedge every later run."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    target = bundled.cache_dir()
    stale = target.with_name(target.name + ".unpacking")
    stale.mkdir(parents=True)
    (stale / "junk").write_text("left over", encoding="utf-8")
    out = bundled.ensure(log=lambda *a: None)
    assert (out / "track.json").is_file()
    assert not stale.exists()


@needs_blob
def test_ensure_replaces_a_cache_without_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    target = bundled.cache_dir()
    target.mkdir(parents=True)
    (target / "not-a-track").write_text("x", encoding="utf-8")
    out = bundled.ensure(log=lambda *a: None)
    assert (out / "track.json").is_file()


def test_the_cache_honours_xdg_cache_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "somewhere"))
    assert str(tmp_path / "somewhere") in str(bundled.cache_dir())


def test_extract_refuses_a_member_that_escapes_its_directory(tmp_path, monkeypatch):
    """Defence against a repacked blob, not against ours.

    The blob is signed with a key that ships, so anybody can repack one. That makes tar path
    traversal a real shape rather than a theoretical one, and an extractor that writes wherever a
    member name says is how a dataset becomes arbitrary file write.
    """
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tar:
        info = tarfile.TarInfo("../escaped")
        data = b"gotcha"
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    blob = bundled.pack(payload.getvalue())
    monkeypatch.setattr(bundled, "_read", lambda: blob)
    with pytest.raises(ValueError, match="outside its own directory"):
        bundled.extract(tmp_path / "x", log=lambda *a: None)


def test_a_blob_with_no_manifest_says_so(monkeypatch):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tar:
        info = tarfile.TarInfo("track/thing")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    monkeypatch.setattr(bundled, "_read", lambda: bundled.pack(payload.getvalue()))
    with pytest.raises(ValueError, match=r"no manifest\.json"):
        bundled.manifest()
