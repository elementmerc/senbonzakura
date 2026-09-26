# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Tests for the corpus builder (`senbonzakura track build`).

The builder exists because two of the three upstream datasets declare no licence, so the rows
cannot be redistributed and only the recipe can. That makes the licence check the load-bearing
part rather than a courtesy, and most of these tests are about it refusing rather than about it
succeeding.
"""
import json
import sys
import urllib.error

import pytest

# Imported as a package module rather than loaded from a path under `tools/`, because that is
# now where it lives: `tools/` ships in no wheel, so the builder the guide sends users to was
# absent from every install.
from senbonzakura import trackbuild as bt


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


# ── which shards a split lives in ────────────────────────────────────────────────
@pytest.mark.parametrize(("files", "want"), [
    # What the Hub's own converter writes, which is what both sources in SOURCES use.
    (["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet",
      "data/test-00000-of-00001.parquet", "README.md"],
     ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]),
    (["train/0000.parquet", "test/0000.parquet"], ["train/0000.parquet"]),
    (["train-00000-of-00001.parquet"], ["train-00000-of-00001.parquet"]),
    (["train.parquet", "test.parquet"], ["train.parquet"]),
])
def test_the_split_shards_are_found_in_every_layout_this_reads(files, want):
    assert bt._split_shards(files, "train") == want


def test_shards_come_back_in_name_order_whatever_order_the_listing_was_in():
    # Name order is the order `datasets` reads them in, and the shard index is in the name, so
    # a listing that arrives shuffled must not become a corpus in a different order from the one
    # every published digest was measured on.
    files = ["data/train-00002-of-00003.parquet", "data/train-00000-of-00003.parquet",
             "data/train-00001-of-00003.parquet"]
    assert bt._split_shards(files, "train") == sorted(files)


def test_an_unmodelled_layout_is_declined_rather_than_guessed_at():
    # None means "hand it to `datasets`", never "there are no rows". A silent empty split here
    # would write a corpus nobody could account for.
    assert bt._split_shards(["data/train.json", "README.md"], "train") is None


def test_a_split_that_is_not_there_is_declined_rather_than_matched_by_prefix():
    assert bt._split_shards(["data/train-00000-of-00001.parquet"], "validation") is None


# ── fetch ────────────────────────────────────────────────────────────────────────
def _write_parquet(path, rows, column="text"):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({column: pa.array(rows, type=pa.string())}), str(path))
    return str(path)


def _with_hub(monkeypatch, tmp_path, shards, *, on_list=None, on_download=None):
    """A `huggingface_hub` holding `shards`, a mapping of repo filename to rows.

    Stubbed at the module rather than at this file's own helpers, so `_repo_files` and
    `_download` are the code under test rather than the code being replaced.
    """
    written = {}
    for i, (name, rows) in enumerate(shards.items()):
        written[name] = _write_parquet(tmp_path / f"shard{i}.parquet", rows) \
            if isinstance(rows, list) else rows
    seen = {}

    def _list(repo, repo_type=None, revision=None):
        seen.update(list_repo=repo, list_type=repo_type, list_revision=revision)
        if on_list is not None:
            on_list()
        return list(shards)

    def _download(repo_id=None, filename=None, repo_type=None, revision=None):
        seen.update(repo=repo_id, filename=filename, type=repo_type, revision=revision)
        if on_download is not None:
            on_download()
        return written[filename]

    mod = type(sys)("huggingface_hub")
    mod.list_repo_files = _list
    mod.hf_hub_download = _download
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return seen


def test_fetch_returns_rows_in_upstream_order(monkeypatch, tmp_path):
    _with_hub(monkeypatch, tmp_path, {"data/train-00000-of-00001.parquet": ["b", "a", "c"]})
    assert bt.fetch(_src(), log=lambda *a: None) == ["b", "a", "c"]


def test_fetch_reads_every_shard_and_keeps_them_in_shard_order(monkeypatch, tmp_path):
    # Two shards concatenated the wrong way round is a corpus that loads, splits and scores, and
    # is not the corpus the manifest describes. It is the one failure this reader must not have.
    _with_hub(monkeypatch, tmp_path, {
        "data/train-00001-of-00002.parquet": ["c", "d"],
        "data/train-00000-of-00002.parquet": ["a", "b"],
    })
    assert bt.fetch(_src(), log=lambda *a: None) == ["a", "b", "c", "d"]


def test_fetch_drops_blank_rows_without_dropping_content(monkeypatch, tmp_path):
    _with_hub(monkeypatch, tmp_path,
              {"data/train-00000-of-00001.parquet": ["a", "   ", "", "b"]})
    assert bt.fetch(_src(), log=lambda *a: None) == ["a", "b"]


def test_fetch_passes_the_pinned_revision_through(monkeypatch, tmp_path):
    seen = _with_hub(monkeypatch, tmp_path, {"data/train-00000-of-00001.parquet": ["a"]})
    bt.fetch(_src(), log=lambda *a: None)
    # Both calls, not one: a listing read at the pin and a download taken from the tip would
    # produce rows nobody could trace to a commit.
    assert seen["list_revision"] == "0" * 40
    assert seen["revision"] == "0" * 40
    assert seen["list_type"] == seen["type"] == "dataset"


def test_fetch_refuses_a_changed_upstream_schema(monkeypatch, tmp_path):
    path = _write_parquet(tmp_path / "s.parquet", ["a"], column="prompt")
    _with_hub(monkeypatch, tmp_path, {"data/train-00000-of-00001.parquet": path})
    with pytest.raises(SystemExit) as e:
        bt.fetch(_src(), log=lambda *a: None)
    assert "no 'text' column" in str(e.value)


def test_fetch_refuses_an_empty_upstream(monkeypatch, tmp_path):
    _with_hub(monkeypatch, tmp_path, {"data/train-00000-of-00001.parquet": ["", "  "]})
    with pytest.raises(SystemExit) as e:
        bt.fetch(_src(), log=lambda *a: None)
    assert "no non-empty prompts" in str(e.value)


def test_fetch_explains_a_deleted_revision_rather_than_raising_a_library_error(
        monkeypatch, tmp_path):
    def _boom():
        raise ValueError("Revision not found")

    _with_hub(monkeypatch, tmp_path, {"data/train-00000-of-00001.parquet": ["a"]}, on_list=_boom)
    with pytest.raises(SystemExit) as e:
        bt.fetch(_src(), log=lambda *a: None)
    assert "re-pinning" in str(e.value)


def test_a_failed_download_names_the_file_rather_than_raising_a_library_error(
        monkeypatch, tmp_path):
    def _boom():
        raise OSError("connection reset")

    _with_hub(monkeypatch, tmp_path, {"data/train-00000-of-00001.parquet": ["a"]},
              on_download=_boom)
    with pytest.raises(SystemExit) as e:
        bt.fetch(_src(), log=lambda *a: None)
    assert "could not download data/train-00000-of-00001.parquet" in str(e.value)


def test_a_shard_that_stops_partway_through_is_refused_rather_than_counted(
        monkeypatch, tmp_path):
    # A shard that opens and then dies mid-read is the worst of the three failures here: the rows
    # already collected look like a pool, and a pool shorter than the one the manifest describes
    # is a corpus nobody can account for afterwards.
    import pyarrow.parquet as pq

    class _Dies:
        def __init__(self, *a, **k):
            self.schema_arrow = type("S", (), {"names": ["text"]})()

        def iter_batches(self, columns=None):
            raise OSError("input/output error")
            yield

    _with_hub(monkeypatch, tmp_path, {"data/train-00000-of-00001.parquet": ["a"]})
    monkeypatch.setattr(pq, "ParquetFile", _Dies)
    with pytest.raises(SystemExit) as e:
        bt.fetch(_src(), log=lambda *a: None)
    assert "could not be read to the end" in str(e.value)


def test_a_shard_that_is_not_parquet_is_a_sentence_rather_than_a_stack_trace(
        monkeypatch, tmp_path):
    half = tmp_path / "half.parquet"
    half.write_bytes(b"PAR1 and then nothing")
    _with_hub(monkeypatch, tmp_path, {"data/train-00000-of-00001.parquet": str(half)})
    with pytest.raises(SystemExit) as e:
        bt.fetch(_src(), log=lambda *a: None)
    assert "not a readable parquet file" in str(e.value)


def test_a_layout_this_reader_declines_says_which_source_needed_datasets(monkeypatch, tmp_path):
    _with_hub(monkeypatch, tmp_path, {"data/train.json": ["a"]})
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(SystemExit) as e:
        bt.fetch(_src(repo="odd/layout"), log=lambda *a: None)
    assert "odd/layout" in str(e.value)
    assert "senbonzakura[hub]" in str(e.value)


def test_a_layout_this_reader_declines_is_read_by_datasets_when_it_is_there(monkeypatch, tmp_path):
    class _FakeDS:
        column_names = ("text",)

        def __getitem__(self, key):
            return ["a", "b"]

    mod = type(sys)("datasets")
    mod.load_dataset = lambda *a, **k: _FakeDS()
    monkeypatch.setitem(sys.modules, "datasets", mod)
    _with_hub(monkeypatch, tmp_path, {"data/train.json": ["a"]})
    assert bt.fetch(_src(), log=lambda *a: None) == ["a", "b"]


def test_the_datasets_fallback_still_refuses_a_changed_schema(monkeypatch, tmp_path):
    class _FakeDS:
        column_names = ("prompt",)

        def __getitem__(self, key):
            return ["a"]

    mod = type(sys)("datasets")
    mod.load_dataset = lambda *a, **k: _FakeDS()
    monkeypatch.setitem(sys.modules, "datasets", mod)
    _with_hub(monkeypatch, tmp_path, {"data/train.json": ["a"]})
    with pytest.raises(SystemExit) as e:
        bt.fetch(_src(), log=lambda *a: None)
    assert "no 'text' column" in str(e.value)


def test_the_datasets_fallback_explains_a_deleted_revision(monkeypatch, tmp_path):
    def _boom(*a, **k):
        raise ValueError("Revision not found")

    mod = type(sys)("datasets")
    mod.load_dataset = _boom
    monkeypatch.setitem(sys.modules, "datasets", mod)
    _with_hub(monkeypatch, tmp_path, {"data/train.json": ["a"]})
    with pytest.raises(SystemExit) as e:
        bt.fetch(_src(), log=lambda *a: None)
    assert "re-pinning" in str(e.value)


def test_a_missing_huggingface_hub_reads_as_a_damaged_install_not_a_missing_extra(monkeypatch):
    # It is a base dependency. Sending this person to an optional extra sends them somewhere
    # that cannot help, which is the mistake `trackio.py` had to correct in its own reader.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(SystemExit) as e:
        bt.fetch(_src(), log=lambda *a: None)
    assert "install is damaged" in str(e.value)


def test_a_missing_pyarrow_reads_as_a_damaged_install_too(monkeypatch, tmp_path):
    _with_hub(monkeypatch, tmp_path, {"data/train-00000-of-00001.parquet": ["a"]})
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(SystemExit) as e:
        bt.fetch(_src(), log=lambda *a: None)
    assert "install is damaged" in str(e.value)


def test_fetch_needs_no_datasets_for_the_sources_this_recipe_actually_pins(monkeypatch, tmp_path):
    # THE DEFECT THIS WHOLE PATH WAS REWRITTEN FOR. `datasets` is the `hub` extra and every
    # documented install string is the bare package, so the second command of the documented
    # first run died on an import. It must stay dead: with `datasets` unimportable, every source
    # in SOURCES still reads.
    monkeypatch.setitem(sys.modules, "datasets", None)
    for i, src in enumerate(bt.SOURCES):
        shard = f"data/{src['split']}-00000-of-00001.parquet"
        here = tmp_path / f"s{i}"
        here.mkdir()
        _with_hub(monkeypatch, here, {shard: ["a", "b"]})
        assert bt.fetch(src, log=lambda *a: None) == ["a", "b"]


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
