# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A menu that describes a different install from the one it is running on.

Two of them did.

  * the corpus menu offered AdvBench as `walledai/AdvBench::train` and told the reader it is
    GATED on the Hub and needs `hf auth login` before it will fetch. The package bundles
    AdvBench's 520 prompts, and `--harmful advbench` reads them off the disk with the network
    unplugged. So a newcomer was sent to authenticate against a service for a file they had
    already installed.
  * the model question defaulted to `Qwen/Qwen3-1.7B`, so pressing Enter through a walk whose
    every screen says "press Enter to take the default" started a download of a model nobody had
    chosen. A nominated model is a recommendation, and nothing here measures one model against
    another.
"""
import pytest

from senbonzakura import corpora, interactive


def test_the_bundled_corpora_are_offered_as_bundled():
    rows = interactive.by_side("harmful")
    advbench = [r for r in rows if r["key"] == "advbench"]
    assert advbench, f"AdvBench is in the wheel and is not on the menu: {[r['key'] for r in rows]}"
    assert advbench[0]["spec"] == "advbench", (
        "the menu points at a Hub id for a corpus this package carries, so choosing it fetches "
        "what is already on the disk")


def test_no_bundled_row_sends_anybody_to_authenticate():
    for side in ("harmful", "harmless"):
        for row in interactive.by_side(side):
            if row["spec"] in corpora.CORPORA:
                text = f"{row['title']} {row['note']}".lower()
                assert "gated" not in text and "auth login" not in text, (
                    f"{row['key']} ships inside the package and its menu row asks for an "
                    f"account: {row['note']}")


@pytest.mark.parametrize("key", interactive._BUNDLED_ON_THE_MENU)
def test_the_menu_quotes_the_corpus_table_rather_than_a_copy_of_it(key):
    """GENERATED, NOT TYPED. The row counts and licences here are the ones the licence obligations
    depend on, and this file already carried a copy of them that had gone wrong in the worst
    available way.
    """
    c = corpora.CORPORA[key]
    side = interactive._ARM_SIDE[c.arm]
    row = next(r for r in interactive.by_side(side) if r["key"] == key)
    assert row["licence"] == c.licence
    assert f"{c.rows:,}" in row["note"]
    assert row["title"] == c.name


def test_something_of_my_own_still_takes_a_bundled_name():
    """The menu is a curated subset, so the corpora left off it have to stay reachable."""
    row = next(r for r in interactive.by_side("harmful") if r["key"] == "own")
    assert "bundled" in row["note"]


# ── the model question ───────────────────────────────────────────────────────────────────

def test_no_model_is_nominated_when_the_machine_has_none(tmp_path, monkeypatch):
    """Enter has to be safe everywhere in this walk, and it was not here."""
    monkeypatch.setattr(interactive, "models_on_this_machine", lambda root=".": [])
    answers = iter(["", "", "mine/own-model"])
    said = []
    got = interactive.pick_model(ask_fn=lambda _p: next(answers), log=said.append, root=str(tmp_path))
    assert got == "mine/own-model", (
        "pressing Enter produced a model id nobody typed, which starts a download of somebody "
        f"else's choosing. Said: {said}")
    assert "Qwen" not in "\n".join(said), "a nominated model is a recommendation with nothing behind it"


def test_what_is_on_this_machine_is_offered_first(tmp_path, monkeypatch):
    monkeypatch.setattr(interactive, "models_on_this_machine",
                        lambda root=".": [{"id": "here/one", "bytes": 2_340_000_000,
                                           "note": "yours, already edited"}])
    said = []
    got = interactive.pick_model(ask_fn=lambda _p: "1", log=said.append, root=str(tmp_path))
    assert got == "here/one"
    text = "\n".join(said)
    assert "2.34 GB" in text, "the size is the fact that decides whether it will fit"
    assert "nothing is downloaded" in text


def test_a_cache_entry_with_no_weights_is_not_offered(monkeypatch):
    """IT IS NOT "IS IT CACHED", IT IS "DOES THE SNAPSHOT HOLD WEIGHTS".

    A repository lands in the cache the moment anything reads its config. The development
    machine's cache holds two repos that are one `config.json` each, and a picker built on the
    index offers a 17 GB model as ready to go and is wrong about both words.
    """
    class _File:
        def __init__(self, name):
            self.file_name = name

    class _Rev:
        def __init__(self, names):
            self.files = [_File(n) for n in names]

    class _Repo:
        def __init__(self, repo_id, names, size):
            self.repo_id, self.repo_type, self.size_on_disk = repo_id, "model", size
            self.revisions = [_Rev(names)]

    class _Info:
        def __init__(self):
            self.repos = [_Repo("config/only", ["config.json"], 17_000_000_000),
                          _Repo("real/weights", ["config.json", "model.safetensors"],
                                2_000_000_000)]

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda *a, **k: _Info())
    ids = [m["id"] for m in interactive.cached_models()]
    assert ids == ["real/weights"], ids


def test_an_unreadable_cache_is_a_reason_to_ask_rather_than_to_stop(monkeypatch):
    import huggingface_hub

    def boom(*_a, **_k):
        raise OSError("no cache here")

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", boom)
    assert interactive.cached_models() == []


def test_a_directory_you_edited_is_a_model_you_can_start_from(tmp_path):
    out = tmp_path / "lfm2-brain"
    out.mkdir()
    (out / "config.json").write_text("{}", encoding="utf-8")
    (out / "model.safetensors").write_bytes(b"0" * 1024)
    (tmp_path / "not-a-model").mkdir()
    found = interactive.edited_models(str(tmp_path))
    assert [m["id"] for m in found] == [str(out)]
    assert found[0]["note"], "a previous output needs its own label, since it is already edited"
