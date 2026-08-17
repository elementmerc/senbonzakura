"""`senbonzakura fetch`, and the three ways a download is wrong.

A byte count catches the first. Nothing catches the third by accident, and the third is the one
that matters for a comparison: a Q8_0 where a Q4_K_M was expected loads without complaint and
silently changes every speed and memory figure it appears in.

The transfer itself is exercised against a real local HTTP server rather than a mocked `urlopen`,
because a mock proves the code calls what it was written to call and nothing about whether bytes
arrive intact.
"""
import hashlib
import http.server
import struct
import threading

import pytest

from senbonzakura import fetch
from senbonzakura.fetch import FetchError

U32, STRING = 4, 8


def _gguf_bytes(*, ftype=15, arch="lfm2", tensors=148):
    """A real GGUF header, assembled to the format specification."""
    def kv_str(k, v):
        kb, vb = k.encode(), v.encode()
        return (struct.pack("<Q", len(kb)) + kb + struct.pack("<I", STRING)
                + struct.pack("<Q", len(vb)) + vb)

    def kv_u32(k, v):
        kb = k.encode()
        return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", U32) + struct.pack("<I", v)

    body = kv_str("general.architecture", arch) + kv_u32("general.file_type", ftype)
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", tensors)
            + struct.pack("<Q", 2) + body + b"\x00" * 256)


# ── reading the source ─────────────────────────────────────────────────────────────
def test_a_hub_reference_splits_on_the_colon():
    assert fetch.parse_source("LiquidAI/LFM2.5-8B-A1B-GGUF:m-Q4_K_M.gguf") == (
        "hub", "LiquidAI/LFM2.5-8B-A1B-GGUF", "m-Q4_K_M.gguf")


def test_an_https_url_yields_its_filename():
    assert fetch.parse_source("https://h.co/a/b/m-Q4_K_M.gguf") == (
        "url", "https://h.co/a/b/m-Q4_K_M.gguf", "m-Q4_K_M.gguf")


def test_a_query_string_is_not_part_of_the_filename():
    _, _, name = fetch.parse_source("https://h.co/m.gguf?download=true")
    assert name == "m.gguf"


def test_plain_http_is_refused_rather_than_upgraded():
    """A model swapped in transit would pass every other check in this module: it would be a
    complete, valid GGUF of the right architecture and quantisation.
    """
    with pytest.raises(FetchError) as e:
        fetch.parse_source("http://h.co/m.gguf")
    assert "plain HTTP" in str(e.value)
    assert "Use https" in str(e.value)


def test_a_windows_path_is_not_read_as_a_hub_reference():
    # C:\models\m.gguf splits on the same colon a Hub id uses. A Hub id always has an owner and
    # no drive letter does, which is the distinction.
    with pytest.raises(FetchError, match="could not read"):
        fetch.parse_source(r"C:\models\m.gguf")


def test_a_repo_with_no_file_says_what_is_missing():
    with pytest.raises(FetchError) as e:
        fetch.parse_source("LiquidAI/LFM2.5-8B-A1B-GGUF:")
    assert "names a repository and no file" in str(e.value)


def test_nonsense_is_refused_with_both_forms_named():
    with pytest.raises(FetchError) as e:
        fetch.parse_source("just-a-word")
    assert "repo_id:filename" in str(e.value)
    assert "https" in str(e.value)


# ── the credential, which must not come from an argument by preference ─────────────
def test_an_explicit_token_wins(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert fetch.resolve_token("explicit") == "explicit"


def test_the_environment_is_used_when_no_argument_is_given(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert fetch.resolve_token() == "from-env"


def test_the_alternative_environment_name_is_honoured(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "other")
    assert fetch.resolve_token() == "other"


def test_no_token_anywhere_defers_to_the_hubs_own_store(monkeypatch):
    """None, not an empty string: huggingface_hub reads its stored login when passed None."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert fetch.resolve_token() is None


# ── verification: the three failures ──────────────────────────────────────────────
def test_a_short_file_is_caught_and_told_why_it_matters(tmp_path):
    p = tmp_path / "m-Q4_K_M.gguf"
    p.write_bytes(_gguf_bytes())
    with pytest.raises(FetchError) as e:
        fetch.verify(p, expect_size=5_000_000_000, log=lambda _m: None)
    msg = str(e.value)
    assert "short by" in msg
    assert "still loads and still serves" in msg


def test_an_error_page_is_caught_by_the_header_not_the_length(tmp_path):
    """It saves at exactly the length the server promised, so only the first bytes give it away."""
    p = tmp_path / "m-Q4_K_M.gguf"
    p.write_bytes(b"<!DOCTYPE html><html>404</html>")
    with pytest.raises(FetchError) as e:
        fetch.verify(p, log=lambda _m: None)
    assert "not the GGUF it should be" in str(e.value)


def test_the_wrong_quantisation_is_caught(tmp_path):
    """THE CASE. Complete, valid, loads, wrong, and invisible to every other check."""
    p = tmp_path / "m-Q4_K_M.gguf"
    p.write_bytes(_gguf_bytes(ftype=7))          # actually Q8_0
    with pytest.raises(FetchError) as e:
        fetch.verify(p, log=lambda _m: None)
    assert "Q8_0" in str(e.value) and "Q4_K_M" in str(e.value)


def test_the_wrong_architecture_is_caught(tmp_path):
    p = tmp_path / "m-Q4_K_M.gguf"
    p.write_bytes(_gguf_bytes(arch="qwen3"))
    with pytest.raises(FetchError, match="different model"):
        fetch.verify(p, expect_arch="lfm2", log=lambda _m: None)


def test_a_matching_file_passes_every_check(tmp_path):
    p = tmp_path / "m-Q4_K_M.gguf"
    body = _gguf_bytes()
    p.write_bytes(body)
    head = fetch.verify(p, expect_size=len(body), expect_arch="lfm2",
                        expect_sha256=hashlib.sha256(body).hexdigest(), log=lambda _m: None)
    assert head["file_type"] == "Q4_K_M"


def test_a_wrong_digest_is_refused(tmp_path):
    p = tmp_path / "m-Q4_K_M.gguf"
    p.write_bytes(_gguf_bytes())
    with pytest.raises(FetchError) as e:
        fetch.verify(p, expect_sha256="0" * 64, log=lambda _m: None)
    assert "do not use this file" in str(e.value)


def test_a_digest_is_compared_case_insensitively(tmp_path):
    p = tmp_path / "m-Q4_K_M.gguf"
    body = _gguf_bytes()
    p.write_bytes(body)
    fetch.verify(p, expect_sha256=hashlib.sha256(body).hexdigest().upper(), log=lambda _m: None)


def test_the_quant_check_can_be_waived_explicitly(tmp_path):
    p = tmp_path / "m-Q4_K_M.gguf"
    p.write_bytes(_gguf_bytes(ftype=7))
    assert fetch.verify(p, expect_quant="none", log=lambda _m: None)["file_type"] == "Q8_0"


def test_the_size_check_runs_before_the_header_is_read(tmp_path):
    """Cheapest first: there is no sense hashing five gigabytes to discover it says HTML."""
    p = tmp_path / "m-Q4_K_M.gguf"
    p.write_bytes(b"<html>not a gguf</html>")
    with pytest.raises(FetchError) as e:
        fetch.verify(p, expect_size=999, log=lambda _m: None)
    # The SIZE complaint, not the header one, because size is checked first.
    assert "bytes and 999 were expected" in str(e.value)


# ── the transfer, against a real server ───────────────────────────────────────────
@pytest.fixture
def server(tmp_path):
    """A real HTTP server over a temp directory. Yields (base_url, root)."""
    root = tmp_path / "srv"
    root.mkdir()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def log_message(self, *a):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_port}", root
    httpd.shutdown()
    httpd.server_close()


def test_a_real_download_arrives_intact(server, tmp_path):
    """The bytes, over a socket. A mocked urlopen proves nothing about a transfer."""
    base, root = server
    body = _gguf_bytes()
    (root / "m-Q4_K_M.gguf").write_bytes(body)

    # No monkeypatching: this goes through the real guard. Loopback over http is allowed on
    # purpose, because there is no network path to loopback to intercept, so the test exercises
    # the shipped code rather than a relaxed copy of it.
    assert fetch.parse_source(f"{base}/m-Q4_K_M.gguf")[0] == "url"
    out = tmp_path / "dl"
    got = fetch.download("url", f"{base}/m-Q4_K_M.gguf", "m-Q4_K_M.gguf", out,
                         log=lambda _m: None)
    assert got.read_bytes() == body
    fetch.verify(got, expect_size=len(body), expect_arch="lfm2", log=lambda _m: None)


def test_a_failed_download_leaves_no_partial_file(server, tmp_path):
    """A `.part` left behind is what a later run mistakes for a finished download."""
    base, _ = server
    out = tmp_path / "dl"
    with pytest.raises(FetchError, match="could not download"):
        fetch.download("url", f"{base}/absent.gguf", "absent.gguf", out, log=lambda _m: None)
    assert not (out / "absent.gguf").exists()
    assert not (out / "absent.gguf.part").exists()


def test_the_download_refuses_a_non_https_reference_at_the_point_of_use(tmp_path):
    """Re-checked here, not only in parse_source: a second caller cannot inherit that guarantee."""
    with pytest.raises(FetchError, match="https is required"):
        fetch.download("url", "ftp://h.co/m.gguf", "m.gguf", tmp_path, log=lambda _m: None)


# ── the command ───────────────────────────────────────────────────────────────────
def test_an_existing_file_is_verified_rather_than_refetched(tmp_path):
    body = _gguf_bytes()
    (tmp_path / "m-Q4_K_M.gguf").write_bytes(body)
    lines = []
    assert fetch.run(["https://h.co/m-Q4_K_M.gguf", "--out", str(tmp_path),
                      "--expect-size", str(len(body))], log=lines.append) == 0
    assert any("already here" in m for m in lines)


def test_a_bad_existing_file_is_kept_for_inspection(tmp_path):
    """Deleting the evidence on the way out is how the same wrong download happens twice."""
    p = tmp_path / "m-Q4_K_M.gguf"
    p.write_bytes(_gguf_bytes(ftype=7))
    with pytest.raises(SystemExit) as e:
        fetch.run(["https://h.co/m-Q4_K_M.gguf", "--out", str(tmp_path)], log=lambda _m: None)
    assert "for inspection" in str(e.value)
    assert p.exists()


def test_a_non_gguf_file_skips_the_header_checks(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert fetch.run(["https://h.co/tokenizer.json", "--out", str(tmp_path)],
                     log=lambda _m: None) == 0
