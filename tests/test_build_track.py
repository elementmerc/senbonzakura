"""Tests for the corpus builder (tools/build_track.py).

The builder exists because two of the three upstream datasets declare no licence, so the rows
cannot be redistributed and only the recipe can. That makes the licence check the load-bearing
part rather than a courtesy, and most of these tests are about it refusing rather than about it
succeeding.
"""
import importlib.util
import json
import sys
import urllib.error
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_track", Path(__file__).resolve().parent.parent / "tools" / "build_track.py")
bt = importlib.util.module_from_spec(_SPEC)
sys.modules["build_track"] = bt
_SPEC.loader.exec_module(bt)


def _src(repo="a/b", side="harmful", licence="apache-2.0"):
    return {"repo": repo, "side": side, "split": "train", "revision": "0" * 40,
            "licence": licence, "note": ""}


# ── the recorded table itself ────────────────────────────────────────────────────
def test_sources_cover_both_sides():
    sides = {s["side"] for s in bt.SOURCES}
    assert sides == {"harmful", "harmless"}


def test_every_source_pins_a_full_revision():
    # A tag or a branch is not a pin: it moves. The recipe is only reproducible against a commit.
    for s in bt.SOURCES:
        assert len(s["revision"]) == 40, s["repo"]
        assert all(c in "0123456789abcdef" for c in s["revision"]), s["repo"]


def test_the_undeclared_licence_is_recorded_as_none_not_omitted():
    # "declares nothing" is the finding, so it has to be a value in the table rather than a
    # missing key that reads as an oversight.
    harmless = [s for s in bt.SOURCES if s["side"] == "harmless"]
    assert harmless and all("licence" in s for s in harmless)
    assert any(s["licence"] is None for s in bt.SOURCES)


def test_citations_name_advbench_and_alpaca():
    assert "AdvBench" in bt.CITATIONS
    assert "Alpaca" in bt.CITATIONS
    assert "CC BY-NC 4.0" in bt.CITATIONS


# ── declared_licence ─────────────────────────────────────────────────────────────
def test_declared_licence_reads_the_card_field(monkeypatch):
    monkeypatch.setattr(bt, "_urlopen_json", None, raising=False)

    class _R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"cardData": {"license": "apache-2.0"}}'

    monkeypatch.setattr(bt.urllib.request, "urlopen", lambda *a, **k: _R())
    assert bt.declared_licence("a/b") == "apache-2.0"


def test_declared_licence_returns_none_when_nothing_is_declared(monkeypatch):
    class _R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"cardData": {"language": ["en"]}}'

    monkeypatch.setattr(bt.urllib.request, "urlopen", lambda *a, **k: _R())
    assert bt.declared_licence("a/b") is None


def test_declared_licence_returns_none_when_there_is_no_card_at_all(monkeypatch):
    class _R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    monkeypatch.setattr(bt.urllib.request, "urlopen", lambda *a, **k: _R())
    assert bt.declared_licence("a/b") is None


def test_declared_licence_refuses_rather_than_guessing_when_the_hub_is_unreachable(monkeypatch):
    def _boom(*a, **k):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(bt.urllib.request, "urlopen", _boom)
    with pytest.raises(SystemExit) as e:
        bt.declared_licence("a/b")
    assert "Refusing to assemble a corpus" in str(e.value)


# ── check_licences ───────────────────────────────────────────────────────────────
def test_check_licences_passes_when_upstream_matches_the_table(monkeypatch):
    monkeypatch.setattr(bt, "declared_licence", lambda repo: "apache-2.0")
    bt.check_licences([_src()], log=lambda *a: None)


def test_check_licences_passes_when_both_sides_agree_that_nothing_is_declared(monkeypatch):
    monkeypatch.setattr(bt, "declared_licence", lambda repo: None)
    bt.check_licences([_src(licence=None)], log=lambda *a: None)


def test_check_licences_stops_when_an_upstream_has_been_relicensed(monkeypatch):
    monkeypatch.setattr(bt, "declared_licence", lambda repo: "cc-by-nc-4.0")
    with pytest.raises(SystemExit) as e:
        bt.check_licences([_src(licence="apache-2.0")], log=lambda *a: None)
    msg = str(e.value)
    assert "recorded apache-2.0, now cc-by-nc-4.0" in msg
    assert "docs/evaluation-track-card.md" in msg


def test_check_licences_stops_when_a_licence_appears_where_there_was_none(monkeypatch):
    # The direction that looks like good news is still drift: it changes what may be redistributed.
    monkeypatch.setattr(bt, "declared_licence", lambda repo: "mit")
    with pytest.raises(SystemExit) as e:
        bt.check_licences([_src(licence=None)], log=lambda *a: None)
    assert "recorded nothing declared, now mit" in str(e.value)


def test_check_licences_reports_every_drifted_source_not_just_the_first(monkeypatch):
    monkeypatch.setattr(bt, "declared_licence", lambda repo: "mit")
    with pytest.raises(SystemExit) as e:
        bt.check_licences([_src(repo="a/one"), _src(repo="a/two")], log=lambda *a: None)
    assert "a/one" in str(e.value)
    assert "a/two" in str(e.value)


def test_skipping_the_check_makes_no_requests_and_says_so(monkeypatch):
    def _boom(repo):
        raise AssertionError("the network was touched despite --skip-licence-check")

    monkeypatch.setattr(bt, "declared_licence", _boom)
    said = []
    bt.check_licences([_src()], skip=True, log=said.append)
    assert any("skipped" in s for s in said)


# ── fetch ────────────────────────────────────────────────────────────────────────
class _FakeDS:
    def __init__(self, rows, column="text"):
        self._rows = rows
        self.column_names = [column]
        self._column = column

    def __getitem__(self, key):
        assert key == self._column
        return self._rows


def _with_loader(monkeypatch, loader):
    mod = type(sys)("datasets")
    mod.load_dataset = loader
    monkeypatch.setitem(sys.modules, "datasets", mod)


def test_fetch_returns_rows_in_upstream_order(monkeypatch):
    _with_loader(monkeypatch, lambda *a, **k: _FakeDS(["b", "a", "c"]))
    assert bt.fetch(_src(), log=lambda *a: None) == ["b", "a", "c"]


def test_fetch_drops_blank_rows_without_dropping_content(monkeypatch):
    _with_loader(monkeypatch, lambda *a, **k: _FakeDS(["a", "   ", "", "b"]))
    assert bt.fetch(_src(), log=lambda *a: None) == ["a", "b"]


def test_fetch_passes_the_pinned_revision_through(monkeypatch):
    seen = {}

    def _load(repo, split=None, revision=None):
        seen.update(repo=repo, split=split, revision=revision)
        return _FakeDS(["a"])

    _with_loader(monkeypatch, _load)
    bt.fetch(_src(), log=lambda *a: None)
    assert seen["revision"] == "0" * 40
    assert seen["split"] == "train"


def test_fetch_refuses_a_changed_upstream_schema(monkeypatch):
    _with_loader(monkeypatch, lambda *a, **k: _FakeDS(["a"], column="prompt"))
    with pytest.raises(SystemExit) as e:
        bt.fetch(_src(), log=lambda *a: None)
    assert "no 'text' column" in str(e.value)


def test_fetch_refuses_an_empty_upstream(monkeypatch):
    _with_loader(monkeypatch, lambda *a, **k: _FakeDS(["", "  "]))
    with pytest.raises(SystemExit) as e:
        bt.fetch(_src(), log=lambda *a: None)
    assert "no non-empty prompts" in str(e.value)


def test_fetch_explains_a_deleted_revision_rather_than_raising_a_library_error(monkeypatch):
    def _load(*a, **k):
        raise ValueError("Revision not found")

    _with_loader(monkeypatch, _load)
    with pytest.raises(SystemExit) as e:
        bt.fetch(_src(), log=lambda *a: None)
    assert "re-pinning" in str(e.value)


def test_fetch_says_what_to_install_when_datasets_is_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(SystemExit) as e:
        bt.fetch(_src(), log=lambda *a: None)
    assert "pip install datasets" in str(e.value)


# ── write_atomic ─────────────────────────────────────────────────────────────────
def test_write_atomic_writes_one_prompt_per_line(tmp_path):
    p = tmp_path / "harmful.txt"
    bt.write_atomic(str(p), ["one", "two"])
    assert p.read_text(encoding="utf-8") == "one\ntwo\n"


def test_write_atomic_flattens_embedded_newlines(tmp_path):
    # One prompt per line is the contract the reader downstream depends on, so a prompt that
    # contains a newline must not silently become two prompts.
    p = tmp_path / "harmful.txt"
    bt.write_atomic(str(p), ["a\nb", "c\r\nd"])
    assert p.read_text(encoding="utf-8").count("\n") == 2


def test_write_atomic_leaves_nothing_behind_when_it_fails(tmp_path, monkeypatch):
    p = tmp_path / "harmful.txt"

    class _Boom:
        def replace(self, *a):
            raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        bt.write_atomic(str(p), [_Boom()])
    assert not p.exists()
    assert not (tmp_path / "harmful.txt.partial").exists()


# ── build ────────────────────────────────────────────────────────────────────────
def _stub_build(monkeypatch, rows=("harmful one", "harmless one")):
    monkeypatch.setattr(bt, "check_licences", lambda *a, **k: None)
    monkeypatch.setattr(
        bt, "fetch",
        lambda src, log=print: [f"{src['side']} {i}" for i in range(3)])


def test_build_writes_both_pools_and_a_manifest(tmp_path, monkeypatch):
    _stub_build(monkeypatch)
    m = bt.build(str(tmp_path), log=lambda *a: None)
    assert (tmp_path / "harmful.txt").exists()
    assert (tmp_path / "harmless.txt").exists()
    assert json.loads((tmp_path / "sources.json").read_text())["schema"] == "senbonzakura-corpus/1"
    assert m["outputs"]["harmful"]["rows"] == 3


def test_the_manifest_records_that_it_does_not_reproduce_the_published_track(tmp_path, monkeypatch):
    # The one claim a reader is most likely to assume and most likely to be wrong about.
    _stub_build(monkeypatch)
    m = bt.build(str(tmp_path), log=lambda *a: None)
    assert m["reproduces_published_track"] is False


def test_the_manifest_hashes_what_was_written(tmp_path, monkeypatch):
    _stub_build(monkeypatch)
    m = bt.build(str(tmp_path), log=lambda *a: None)
    assert m["outputs"]["harmful"]["sha256"] == bt.sha256_of(str(tmp_path / "harmful.txt"))


def test_the_manifest_records_a_skipped_licence_check(tmp_path, monkeypatch):
    _stub_build(monkeypatch)
    m = bt.build(str(tmp_path), skip_licence_check=True, log=lambda *a: None)
    assert m["licence_check_skipped"] is True


def test_build_refuses_when_one_side_came_back_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "check_licences", lambda *a, **k: None)
    monkeypatch.setattr(
        bt, "fetch",
        lambda src, log=print: [] if src["side"] == "harmless" else ["a"])
    with pytest.raises(SystemExit) as e:
        bt.build(str(tmp_path), log=lambda *a: None)
    assert "needs both sides" in str(e.value)


def test_build_prints_the_citations_and_the_reproducibility_warning(tmp_path, monkeypatch):
    _stub_build(monkeypatch)
    said = []
    bt.build(str(tmp_path), log=said.append)
    out = "\n".join(said)
    assert "AdvBench" in out
    assert "cannot be rebuilt by anybody" in out
    assert "senbonzakura track" in out


def test_build_never_prints_a_prompt(tmp_path, monkeypatch):
    # Its output has to be safe to paste into an issue, which is the same rule the track builder
    # holds itself to.
    monkeypatch.setattr(bt, "check_licences", lambda *a, **k: None)
    monkeypatch.setattr(bt, "fetch", lambda src, log=print: ["SECRETPROMPTTEXT"] * 3)
    said = []
    bt.build(str(tmp_path), log=said.append)
    assert "SECRETPROMPTTEXT" not in "\n".join(said)


# ── the parser ───────────────────────────────────────────────────────────────────
def test_out_is_required():
    with pytest.raises(SystemExit):
        bt.build_parser().parse_args([])


def test_main_builds_into_the_requested_directory(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(bt, "build", lambda out, **k: seen.update(out=out, **k))
    assert bt.main(["--out", str(tmp_path), "--skip-licence-check"]) == 0
    assert seen["out"] == str(tmp_path)
    assert seen["skip_licence_check"] is True
