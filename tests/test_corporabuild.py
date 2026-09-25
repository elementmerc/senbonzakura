# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The corpus builder's own logic, which nothing exercised until now.

WHY THIS FILE EXISTS

`senbonzakura corpora` fetches six prompt sets at pinned commits, verifies them, packs them into
the blob the wheel ships, and reads that blob back. It is the release path for the harmful-prompt
corpora, and when it moved into the package on 2026-09-23 its coverage was measured for the first
time: 30%, with the whole of `build()` untouched. It had lived under `tools/`, which coverage
never measured, so the number had never been visible rather than having fallen.

WHAT IS WORTH ASSERTING HERE

Not the fetching, which is somebody else's network. The refusals: a file at a pinned commit whose
bytes have changed, a pack that does not read back as what was written, and the difference
between a checkout and an install. Those are the properties that decide whether the corpora a
user measures on are the corpora we think they are.

Nothing here touches the network. `fetch` is replaced throughout.
"""
from __future__ import annotations

import json

import pytest

from senbonzakura import corpora, corporabuild


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point every path the builder writes to at a temporary directory."""
    pkg = tmp_path / "pkg"
    (pkg / "data").mkdir(parents=True)
    (pkg / "vendor").mkdir(parents=True)
    monkeypatch.setattr(corporabuild, "PKG", pkg)
    monkeypatch.setattr(corporabuild, "PINS", pkg / "vendor" / "corpora-pins.json")
    monkeypatch.setattr(corporabuild, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(corporabuild, "ROOT", tmp_path / "root")
    # An install by default: the notices file belongs to a checkout and is asserted separately.
    monkeypatch.setattr(corporabuild, "in_a_checkout", lambda: False)
    return tmp_path


@pytest.fixture
def fake_upstreams(monkeypatch):
    """One deterministic byte string per distinct upstream file, and no network."""
    payloads = {}
    calls = []

    def _fetch(repo, commit, path, **_kw):
        calls.append((repo, commit, path))
        return payloads.setdefault(f"{repo}@{commit}:{path}",
                                   f"prompt\nrow for {repo} {path}\n".encode())

    monkeypatch.setattr(corporabuild, "fetch", _fetch)
    # The shape checks belong to `corpora` and have their own tests; here they must not be the
    # thing under test, or every assertion below becomes an assertion about CSV parsing.
    monkeypatch.setattr(corpora, "extract", lambda _raw, c: [f"p{i}" for i in range(c.rows)])
    monkeypatch.setattr(corpora, "check_count", lambda _c, _prompts: None)
    return calls


# ── one fetch per distinct upstream file ─────────────────────────────────────────

def test_a_file_backing_two_corpora_is_fetched_once(sandbox, fake_upstreams, monkeypatch):
    """XSTest backs two corpora and HarmBench backs two more. Fetching per corpus would double
    the requests and could, in principle, retrieve two different copies of one file and never
    notice, which is the case the keying exists for.
    """
    holder = {}
    monkeypatch.setattr(corpora, "load", lambda key: holder["payload"][key])
    holder["payload"] = corporabuild.build(check_only=True, log=lambda _m: None)

    distinct = {corporabuild.source_key(c) for c in corpora.CORPORA.values()}
    assert len(fake_upstreams) == len(distinct), (
        f"{len(fake_upstreams)} fetches for {len(distinct)} distinct files")
    assert len(distinct) < len(corpora.CORPORA), (
        "this test asserts nothing unless at least one file really does back two corpora")


def test_check_only_writes_nothing(sandbox, fake_upstreams):
    corporabuild.build(check_only=True, log=lambda _m: None)
    assert not corporabuild.PINS.exists(), "a check wrote the pins file"
    assert not (corporabuild.PKG / "data" / corpora.CORPORA_BLOB).exists(), "a check packed a blob"


# ── the hash, which is the whole point of a pinned commit ────────────────────────

def test_a_first_fetch_records_the_hash_it_received(sandbox, fake_upstreams, monkeypatch):
    holder = {}
    monkeypatch.setattr(corpora, "load", lambda key: holder["payload"][key])
    holder["payload"] = corporabuild.build(check_only=True, log=lambda _m: None)
    corporabuild.build(log=lambda _m: None)

    recorded = json.loads(corporabuild.PINS.read_text())["files"]
    assert recorded, "nothing was recorded"
    for key, entry in recorded.items():
        assert len(entry["sha256"]) == 64, f"{key} recorded no usable digest"
        assert entry["bytes"] > 0


def test_a_changed_file_at_a_pinned_commit_is_refused(sandbox, fake_upstreams, monkeypatch):
    """A file at a pinned commit cannot legitimately change. This is the one refusal that says
    somebody has moved something underneath us, so it must stop before anything is packed.
    """
    holder = {}
    monkeypatch.setattr(corpora, "load", lambda key: holder["payload"][key])
    holder["payload"] = corporabuild.build(check_only=True, log=lambda _m: None)
    corporabuild.build(log=lambda _m: None)

    pins = json.loads(corporabuild.PINS.read_text())
    key = next(iter(pins["files"]))
    pins["files"][key]["sha256"] = "0" * 64
    corporabuild.PINS.write_text(json.dumps(pins))
    # The cache would satisfy the read, so the mismatch is the only thing that can fire.
    with pytest.raises(corporabuild.BuildError) as e:
        corporabuild.build(log=lambda _m: None)
    msg = str(e.value)
    assert "PINNED COMMIT" in msg
    assert "Nothing has been packed" in msg, "the message must say the build stopped in time"


def test_a_matching_hash_passes_quietly(sandbox, fake_upstreams, monkeypatch):
    holder = {}
    monkeypatch.setattr(corpora, "load", lambda key: holder["payload"][key])
    holder["payload"] = corporabuild.build(check_only=True, log=lambda _m: None)
    corporabuild.build(log=lambda _m: None)
    said = []
    corporabuild.build(log=said.append)
    assert any("matches the recorded value" in line for line in said)


# ── the cache, so a rebuild does not re-fetch ────────────────────────────────────

def test_the_second_build_reads_the_local_copy(sandbox, fake_upstreams, monkeypatch):
    holder = {}
    monkeypatch.setattr(corpora, "load", lambda key: holder["payload"][key])
    holder["payload"] = corporabuild.build(check_only=True, log=lambda _m: None)
    first = len(fake_upstreams)
    said = []
    corporabuild.build(check_only=True, log=said.append)
    assert len(fake_upstreams) == first, "the second build went back to the network"
    assert any("using the local copy" in line for line in said)


# ── the round trip, which is the difference between writing and shipping ─────────

def test_a_pack_that_does_not_read_back_is_refused(sandbox, fake_upstreams, monkeypatch):
    """Writing a pack is not the same claim as shipping a usable one, and the round trip is the
    only thing that tells them apart.
    """
    monkeypatch.setattr(corpora, "load", lambda _key: ["something else entirely"])
    with pytest.raises(corporabuild.BuildError, match="did not read back"):
        corporabuild.build(log=lambda _m: None)


def test_a_good_pack_is_written_and_round_trips(sandbox, fake_upstreams, monkeypatch):
    holder = {}
    monkeypatch.setattr(corpora, "load", lambda key: holder["payload"][key])
    holder["payload"] = corporabuild.build(check_only=True, log=lambda _m: None)
    payload = corporabuild.build(log=lambda _m: None)

    blob = corporabuild.PKG / "data" / corpora.CORPORA_BLOB
    assert blob.is_file() and blob.stat().st_size > 0
    assert set(payload) == set(corpora.CORPORA)


# ── a checkout and an install are different places ───────────────────────────────

def test_an_install_does_not_write_the_notices_file(sandbox, fake_upstreams, monkeypatch):
    """Writing a Markdown file two levels above site-packages puts it where nobody looks."""
    holder = {}
    monkeypatch.setattr(corpora, "load", lambda key: holder["payload"][key])
    holder["payload"] = corporabuild.build(check_only=True, log=lambda _m: None)
    said = []
    corporabuild.build(log=said.append)
    assert not (corporabuild.ROOT / "THIRD-PARTY-CORPORA.md").exists()
    assert any("not a checkout" in line for line in said), (
        "the skip has to be announced, or a maintainer wonders where the file went")


def test_a_checkout_writes_the_notices_beside_the_repository(sandbox, fake_upstreams, monkeypatch):
    monkeypatch.setattr(corporabuild, "in_a_checkout", lambda: True)
    corporabuild.ROOT.mkdir(parents=True, exist_ok=True)
    holder = {}
    monkeypatch.setattr(corpora, "load", lambda key: holder["payload"][key])
    holder["payload"] = corporabuild.build(check_only=True, log=lambda _m: None)
    corporabuild.build(log=lambda _m: None)

    notices = corporabuild.ROOT / "THIRD-PARTY-CORPORA.md"
    assert notices.is_file()
    text = notices.read_text(encoding="utf-8")
    assert "Bundled corpora" in text
    assert "licence" in text.lower(), "the notice has to carry the terms it exists to carry"


# ── the pins file itself ─────────────────────────────────────────────────────────

def test_a_missing_pins_file_is_an_empty_manifest_not_a_crash(sandbox):
    assert corporabuild.read_pins()["files"] == {}


def test_a_corrupt_pins_file_is_refused_in_words(sandbox):
    corporabuild.PINS.write_text("{not json")
    with pytest.raises(corporabuild.BuildError, match="not valid JSON"):
        corporabuild.read_pins()


def test_two_corpora_sharing_a_file_share_a_key():
    by_key = {}
    for name, c in corpora.CORPORA.items():
        by_key.setdefault(corporabuild.source_key(c), []).append(name)
    assert any(len(v) > 1 for v in by_key.values()), (
        "no upstream file backs two corpora, so the keying this asserts is untested")
