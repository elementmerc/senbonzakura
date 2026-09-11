# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The Hub transfer, the `fetch` command, and the bundled-alias branch of the manifest.

Fourth and last batch of shortlist item A. Everything here is a wrapper: `fetch.run` around
`download` and `verify`, `download`'s Hub half around `huggingface_hub`, `_bake_saved_config`
around `_bake_and_save`. Wrappers are where the interesting decisions live even though they hold
almost no arithmetic, because a wrapper decides what happens when the thing it wraps says no.

The clearest one: `fetch.run` KEEPS a file that failed verification. Deleting the evidence on the
way out is how the same wrong download happens twice, and the message names the path so the file
can be looked at. That is a deliberate choice and it had no test.

CPU only, no network: `huggingface_hub` is driven through a fake module.
"""
import json
import sys
import types

import pytest

from senbonzakura import fetch, track

# ── fetch.download, the Hub half ─────────────────────────────────────────────────────────────


class _HfHubHTTPError(Exception):
    pass


def _fake_hub(monkeypatch, downloader):
    """Install a fake `huggingface_hub` and its errors submodule for the duration of a test.

    The real package owns the transfer: it resumes, it checks its own etag, and it applies a
    stored login without the token passing through our code. That is why this is faked rather
    than reimplemented, and why the only thing worth testing here is how its refusals are dressed.
    """
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = downloader
    errors = types.ModuleType("huggingface_hub.errors")
    errors.HfHubHTTPError = _HfHubHTTPError
    hub.errors = errors
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)


def test_a_hub_download_returns_the_path_the_hub_wrote(tmp_path, monkeypatch):
    written = tmp_path / "model.gguf"
    written.write_bytes(b"weights")

    def _dl(repo_id, filename, revision, local_dir, token):
        assert repo_id == "owner/repo" and filename == "model.gguf"
        assert revision == "abc123", "a pinned revision must be passed through, not dropped"
        return str(written)

    _fake_hub(monkeypatch, _dl)
    got = fetch.download("hub", "owner/repo", "model.gguf", tmp_path,
                         revision="abc123", token=None, log=lambda *_a: None)
    assert got == written


def test_a_gated_repository_is_explained_rather_than_raised_raw(tmp_path, monkeypatch):
    """THE MESSAGE MATTERS MORE THAN THE FAILURE.

    A 401 from the Hub means "you need a token" far more often than it means "this does not
    exist", and the remedy names the two ways to supply one. It says NOT to pass a token as an
    argument, which is the whole point: a token on a command line is in the shell history and in
    every process listing on the machine.
    """
    def _dl(**_k):
        raise _HfHubHTTPError("401 Client Error")

    _fake_hub(monkeypatch, _dl)
    with pytest.raises(fetch.FetchError) as e:
        fetch.download("hub", "owner/repo", "m.gguf", tmp_path, log=lambda *_a: None)
    msg = str(e.value)
    assert "gated or private" in msg
    assert "$HF_TOKEN" in msg and "hf auth login" in msg
    assert "rather than passing one as an argument" in msg


@pytest.mark.parametrize("boom", [OSError("disk full"), ValueError("bad repo id")])
def test_a_local_or_argument_failure_is_wrapped_too(boom, tmp_path, monkeypatch):
    """The second except clause. Without it these escape as themselves, from inside a library
    the user did not know was involved.
    """
    def _dl(**_k):
        raise boom

    _fake_hub(monkeypatch, _dl)
    with pytest.raises(fetch.FetchError, match="could not fetch"):
        fetch.download("hub", "owner/repo", "m.gguf", tmp_path, log=lambda *_a: None)


# ── fetch.run: the command around it ─────────────────────────────────────────────────────────

def test_an_unparseable_source_exits_without_a_traceback(tmp_path):
    with pytest.raises(SystemExit):
        fetch.run(["not a source at all", "--out", str(tmp_path)], log=lambda *_a: None)


def test_a_file_already_here_is_verified_rather_than_downloaded_again(tmp_path, monkeypatch):
    """The cache branch. Re-downloading a file that is already correct wastes the user's evening
    on a large GGUF, so the default is to verify what is there and say so.
    """
    dest = tmp_path / "m.gguf"
    dest.write_bytes(b"weights")

    def _no(*_a, **_k):
        raise AssertionError("an existing file must not be re-downloaded without --force")

    monkeypatch.setattr(fetch, "download", _no)
    monkeypatch.setattr(fetch, "verify", lambda *a, **k: None)

    said = []
    assert fetch.run(["https://example.com/m.gguf", "--out", str(tmp_path)],
                     log=said.append) == 0
    assert any("already here" in s for s in said), "the reader must be told why nothing downloaded"
    assert any("--force" in s for s in said), "and how to override it"


def test_force_re_fetches_even_when_the_file_is_here(tmp_path, monkeypatch):
    dest = tmp_path / "m.gguf"
    dest.write_bytes(b"old")
    called = []

    monkeypatch.setattr(fetch, "download",
                        lambda *a, **k: called.append(True) or dest)
    monkeypatch.setattr(fetch, "verify", lambda *a, **k: None)
    assert fetch.run(["https://example.com/m.gguf", "--out", str(tmp_path), "--force"],
                     log=lambda *_a: None) == 0
    assert called, "--force must actually re-fetch"


def test_a_download_failure_exits_with_the_reason(tmp_path, monkeypatch):
    def _boom(*_a, **_k):
        raise fetch.FetchError("the server hung up")

    monkeypatch.setattr(fetch, "download", _boom)
    with pytest.raises(SystemExit, match="hung up"):
        fetch.run(["https://example.com/m.gguf", "--out", str(tmp_path)], log=lambda *_a: None)


def test_a_file_that_fails_verification_is_kept_and_its_path_named(tmp_path, monkeypatch):
    """THE DELIBERATE CHOICE, and it had no test.

    What is wrong with the file is the interesting part, and deleting the evidence on the way out
    is how the same wrong download happens twice. So the file stays and the message says where.
    """
    dest = tmp_path / "m.gguf"

    def _dl(*_a, **_k):
        dest.write_bytes(b"the wrong bytes")
        return dest

    monkeypatch.setattr(fetch, "download", _dl)

    def _bad(*_a, **_k):
        raise fetch.FetchError("sha256 does not match")

    monkeypatch.setattr(fetch, "verify", _bad)
    with pytest.raises(SystemExit) as e:
        fetch.run(["https://example.com/m.gguf", "--out", str(tmp_path)], log=lambda *_a: None)

    assert "sha256 does not match" in str(e.value)
    assert str(dest) in str(e.value), "the reader cannot inspect a file they cannot find"
    assert dest.exists(), "the evidence was deleted, which is how the same bad fetch repeats"


def test_a_verified_file_reports_success(tmp_path, monkeypatch):
    dest = tmp_path / "m.gguf"
    monkeypatch.setattr(fetch, "download",
                        lambda *a, **k: (dest.write_bytes(b"ok"), dest)[1])
    monkeypatch.setattr(fetch, "verify", lambda *a, **k: None)
    said = []
    assert fetch.run(["https://example.com/m.gguf", "--out", str(tmp_path)], log=said.append) == 0
    assert any("verified" in s for s in said)


# ── track.read_manifest: the `default` alias, which must not read as "no boundaries" ─────────

def test_the_bundled_alias_resolves_before_the_manifest_is_looked_for(tmp_path, monkeypatch):
    """THE BOUNDARY CHECK WOULD OTHERWISE SILENTLY SWITCH OFF for the corpus most people run.

    The literal path "default/track.json" does not exist, and a missing manifest is a legal state
    meaning "boundaries unknown". So without this branch the bundled track, which HAS recorded
    boundaries, would be read as though it had none.
    """
    doc = {"counts": {"bad_ds": 3}, "schema": min(track.KNOWN_SCHEMAS)}
    (tmp_path / "track.json").write_text(json.dumps(doc), encoding="utf-8")

    from senbonzakura import bundled
    monkeypatch.setattr(bundled, "ensure", lambda: tmp_path)
    assert track.read_manifest(track.dataset.BUNDLED_ALIAS) == doc


def test_a_bundled_track_that_cannot_be_unpacked_stops_the_run(monkeypatch):
    """The error is the packaging one from `bundled.ensure`, passed through rather than reduced
    to "no manifest". Reading it as "boundaries unknown" would carry on and measure the wrong rows.
    """
    from senbonzakura import bundled

    def _boom():
        raise bundled.BundledTrackError("this installation was built without its track")

    monkeypatch.setattr(bundled, "ensure", _boom)
    with pytest.raises(SystemExit, match="built without its track"):
        track.read_manifest(track.dataset.BUNDLED_ALIAS)


# ── cli._bake_saved_config: reproducing a winner without paying for the search again ─────────

def test_a_saved_config_is_baked_without_searching(tmp_path):
    """It exists because a crashed save used to cost the whole search a second time.

    Driven through a stand-in rather than a real Abliterator: the method reads two attributes and
    calls one other method, so a real one would add a model load and prove nothing extra.
    """
    from senbonzakura import cli
    from senbonzakura.crashsafe import config_to_bake_args

    cfg = {
        "o_profile": [0.9, 0.5, 0.1, 0.2],
        "d_profile": [0.8, 0.4, 0.0, 0.3],
        "num_directions": 2,
        "dir_mode": "per_layer",
        "direction_index": 0.0,
    }
    path = tmp_path / "winner.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    said, baked = [], []

    class _Args:
        bake_config = str(path)

    class _Stub:
        args = _Args()
        log = staticmethod(said.append)
        _bake_saved_config = cli.Abliterator._bake_saved_config

        def _bake_and_save(self, *a):
            baked.append(a)
            return 0

    assert _Stub()._bake_saved_config("base") == 0
    assert baked, "the saved winner was never baked"
    assert baked[0][:4] == config_to_bake_args(cfg), (
        "the bake arguments must come from the saved file rather than from defaults")
    assert baked[0][4] == "base"
    assert any("skipping the search" in s for s in said), (
        "the run has to say it is not searching, or a reader cannot tell this run from one that "
        "searched and happened to land on the same answer")


# ── doctor.check_quantize: telling a missing library apart from a wrong binary ───────────────

def test_a_quantiser_that_is_not_vendored_is_reported_as_unavailable(monkeypatch):
    from senbonzakura import doctor
    from senbonzakura.vendored import VendorError

    def _boom(*_a, **_k):
        raise VendorError("not vendored in an installed wheel. Build it with tools/vendor_llama.py")

    monkeypatch.setattr(doctor, "find_binary", _boom, raising=False)
    monkeypatch.setattr("senbonzakura.vendored.find_binary", _boom)
    got = doctor.check_quantize()
    assert got.status == "fail"
    assert got.detail == "not available"


def _quantize_saying(monkeypatch, output, exe="/x/llama-quantize"):
    from senbonzakura import doctor
    monkeypatch.setattr("senbonzakura.vendored.find_binary",
                        lambda *_a, **_k: (exe, "vendored"))

    class _R:
        stdout = output
        stderr = b""

    monkeypatch.setattr(doctor.subprocess, "run", lambda *a, **k: _R)
    return doctor.check_quantize()


def test_a_binary_that_will_not_execute_at_all_says_re_vendor(monkeypatch):
    from senbonzakura import doctor
    monkeypatch.setattr("senbonzakura.vendored.find_binary",
                        lambda *_a, **_k: ("/x/llama-quantize", "vendored"))

    def _boom(*_a, **_k):
        raise OSError("exec format error")

    monkeypatch.setattr(doctor.subprocess, "run", _boom)
    got = doctor.check_quantize()
    assert got.status == "fail"
    assert "will not run" in got.detail
    assert "vendor_llama.py" in got.fix


def test_a_missing_shared_library_is_not_reported_as_a_wrong_binary(monkeypatch):
    """MISDIAGNOSED TWICE FOR REAL, inside `python:3.13-slim`, and it happened again today.

    llama.cpp links against OpenMP and a slim image does not carry it. The binary is exactly what
    we think it is and the system is missing a dependency of it, so telling the reader to
    re-vendor sends them to replace a file that is already correct. The remedy has to name the
    library and the package, and has to say that re-vendoring will not help.
    """
    got = _quantize_saying(
        monkeypatch,
        b"/x/llama-quantize: error while loading shared libraries: libgomp.so.1: "
        b"cannot open shared object file: No such file or directory\n")
    assert got.status == "fail"
    assert "cannot start" in got.detail and "libgomp.so.1" in got.detail
    assert "libgomp1" in got.fix
    assert "Re-vendoring will not help" in got.fix


def test_a_quantiser_that_prints_its_usage_passes(monkeypatch):
    got = _quantize_saying(monkeypatch, b"usage: llama-quantize [options] model-f32.gguf\n")
    assert got.status == "pass"


# ── dataset.resolve_pairs: the two refusals that name what is wrong ──────────────────────────

def test_a_plain_text_file_is_refused_for_a_graded_benchmark(tmp_path):
    """One column cannot answer a two-column question, and the message says which shape to use
    rather than failing somewhere downstream with a column that is not there.
    """
    from senbonzakura import dataset
    p = tmp_path / "prompts.txt"
    p.write_text("a\nb\n", encoding="utf-8")
    with pytest.raises(dataset.DatasetError, match="needs two"):
        dataset.resolve_pairs(str(p))


def test_a_path_that_is_neither_a_file_nor_a_hub_id_is_refused(tmp_path):
    """Guessing which of the two the user meant is how a typo becomes a network call."""
    from senbonzakura import dataset
    with pytest.raises(dataset.DatasetError, match="not a Hub id"):
        dataset.resolve_pairs(str(tmp_path / "nothing here"))
