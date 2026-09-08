# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The corpora that ship inside the wheel.

The tool runs with no network once installed, so these are packed rather than fetched. What the
tests guard is not the packing but the two ways a bundled corpus can be quietly wrong: the wrong
rows selected out of a file that parses fine, and two sets merged that measure opposite things.
"""
import csv
import io
import json

import pytest

from senbonzakura import corpora, dataset
from senbonzakura.corpora import CorpusError


def _csv(rows, cols):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


# ── the registry is a set of claims, and each has to be true ──────────────────────
def test_every_corpus_declares_what_a_redistribution_needs():
    # MIT and CC-BY both require the notice to travel with the work, so a corpus without an
    # attribution is a licence breach waiting to ship.
    for key, c in corpora.CORPORA.items():
        assert c.licence, f"{key} declares no licence"
        assert c.attribution, f"{key} names no attribution"
        assert c.upstream and c.commit, f"{key} is not pinned"
        assert c.rows > 0, f"{key} claims no rows"


def test_every_corpus_says_which_arm_it_is():
    """A corpus used as the wrong arm inverts every number taken through it, silently."""
    for key, c in corpora.CORPORA.items():
        assert c.arm in ("harmful", "benign"), f"{key} has arm {c.arm!r}"


def test_the_pins_are_full_commit_shas_not_branches():
    """A branch name resolves to different bytes on different days."""
    for key, c in corpora.CORPORA.items():
        assert len(c.commit) >= 12, f"{key} is pinned to {c.commit!r}, which is not a commit"
        assert c.commit not in ("main", "master", "HEAD"), key


def test_there_is_at_least_one_benign_corpus():
    """Four sets measure whether refusal was removed. Without a benign arm nothing measures the
    other direction, which is the cost side of the same edit.
    """
    assert any(c.arm == "benign" for c in corpora.CORPORA.values())


# ── naming: the ambiguity that matters ────────────────────────────────────────────
def test_a_known_name_resolves():
    assert corpora.resolve_name("advbench") == "advbench"
    assert corpora.resolve_name("  AdvBench  ") == "advbench"


def test_xstest_alone_is_refused_because_merging_it_means_nothing():
    """250 prompts a model SHOULD answer plus 200 it should not, in one list called harmful,
    produces a refusal rate that answers no question at all.
    """
    with pytest.raises(CorpusError) as e:
        corpora.resolve_name("xstest")
    assert "xstest-safe" in str(e.value)
    assert "xstest-unsafe" in str(e.value)


def test_harmbench_all_is_refused_for_the_same_reason():
    with pytest.raises(CorpusError, match="harmbench-copyright"):
        corpora.resolve_name("harmbench-all")


def test_an_unknown_name_lists_what_there_is():
    with pytest.raises(CorpusError) as e:
        corpora.resolve_name("advbenchh")
    assert "advbench" in str(e.value)


# ── extraction: the selection is the part that can be silently wrong ──────────────
def test_a_filter_selects_only_the_rows_it_names():
    raw = _csv([{"Behavior": "a", "FunctionalCategory": "standard"},
                {"Behavior": "b", "FunctionalCategory": "copyright"},
                {"Behavior": "c", "FunctionalCategory": "contextual"},
                {"Behavior": "d", "FunctionalCategory": "standard"}],
               ["Behavior", "FunctionalCategory"])
    got = corpora.extract(raw, corpora.CORPORA["harmbench"])
    assert got == ["a", "d"]


def test_the_other_half_of_a_shared_file_is_reachable():
    raw = _csv([{"prompt": "safe one", "label": "safe"},
                {"prompt": "unsafe one", "label": "unsafe"}], ["prompt", "label"])
    assert corpora.extract(raw, corpora.CORPORA["xstest-safe"]) == ["safe one"]
    assert corpora.extract(raw, corpora.CORPORA["xstest-unsafe"]) == ["unsafe one"]


def test_duplicates_and_blanks_are_dropped():
    raw = _csv([{"goal": "a"}, {"goal": "a"}, {"goal": "  "}, {"goal": "b"}], ["goal"])
    assert corpora.extract(raw, corpora.CORPORA["advbench"]) == ["a", "b"]


def test_order_is_preserved():
    """A partition is taken by slicing, so a corpus that arrives in a different order puts
    different rows in `measure` and two runs that should be comparable are not.
    """
    raw = _csv([{"goal": x} for x in "dcba"], ["goal"])
    assert corpora.extract(raw, corpora.CORPORA["advbench"]) == ["d", "c", "b", "a"]


def test_a_renamed_column_is_a_loud_failure_not_an_empty_corpus():
    """An upstream rename produces a file that parses into nothing, and zero prompts is not a
    failure any later step reports usefully.
    """
    raw = _csv([{"behaviour": "a"}], ["behaviour"])
    with pytest.raises(CorpusError) as e:
        corpora.extract(raw, corpora.CORPORA["harmbench"])
    assert "upstream layout has changed" in str(e.value)


def test_a_byte_order_mark_does_not_break_the_first_column():
    raw = b"\xef\xbb\xbf" + _csv([{"goal": "a"}], ["goal"])
    assert corpora.extract(raw, corpora.CORPORA["advbench"]) == ["a"]


# ── the count check, which catches a filter that quietly matches something else ──
def test_a_changed_count_is_refused():
    c = corpora.CORPORA["advbench"]
    with pytest.raises(CorpusError) as e:
        corpora.check_count(c, ["only", "two"])
    assert "extracted 2" in str(e.value)
    assert "520" in str(e.value)
    assert "Do not ship this" in str(e.value)


def test_the_right_count_passes_through():
    c = corpora.CORPORA["strongreject"]
    assert corpora.check_count(c, ["x"] * c.rows)


# ── the notices, which are a licence obligation rather than a courtesy ───────────
def test_the_notices_name_every_bundled_corpus():
    text = corpora.notices()
    for key, c in corpora.CORPORA.items():
        assert key in text
        assert c.attribution.split(",")[0] in text
        assert c.licence in text


def test_the_notices_are_generated_from_the_same_table_the_loader_reads():
    # Not a hand-maintained file: it cannot fall out of step with what actually ships.
    assert corpora.notices().count("[") == len(corpora.CORPORA)


# ── loading from the pack that ships ─────────────────────────────────────────────
def _packed():
    try:
        corpora.load("advbench")
    except CorpusError:
        return False
    return True


needs_pack = pytest.mark.skipif(
    not _packed(), reason="corpora are not packed; run tools/build_corpora.py")


@needs_pack
@pytest.mark.parametrize("key", sorted(corpora.CORPORA))
def test_every_bundled_corpus_loads_at_its_declared_size(key):
    assert len(corpora.load(key)) == corpora.CORPORA[key].rows


@needs_pack
def test_the_prompts_are_real_text():
    rows = corpora.load("advbench")
    assert all(isinstance(r, str) and r.strip() for r in rows)


@needs_pack
def test_xstests_two_halves_do_not_overlap():
    """They are minimal contrasts of each other, so they are close. They must not be the same."""
    safe = set(corpora.load("xstest-safe"))
    unsafe = set(corpora.load("xstest-unsafe"))
    assert not (safe & unsafe)


@needs_pack
def test_harmbench_and_its_copyright_half_are_disjoint():
    assert not (set(corpora.load("harmbench")) & set(corpora.load("harmbench-copyright")))


def test_an_unpacked_install_says_which_step_was_missed(tmp_path):
    """A source checkout has no pack. That is a different problem from an empty corpus and only
    one of the two is the user's fault.
    """
    with pytest.raises(CorpusError) as e:
        corpora.load("advbench", root=tmp_path)
    assert "build_corpora.py" in str(e.value)


def test_a_pack_that_disagrees_with_the_registry_is_refused(tmp_path, monkeypatch):
    """The names line up and the contents do not: a pack built by a different version."""
    from senbonzakura import bundled
    blob = tmp_path / corpora.CORPORA_BLOB
    doc = {"schema": "senbonzakura-corpora/1", "corpora": {"advbench": ["only one"]}}
    blob.write_bytes(bundled.pack(json.dumps(doc).encode("utf-8")))
    with pytest.raises(CorpusError) as e:
        corpora.load("advbench", root=tmp_path)
    assert "pack and the code disagree" in str(e.value)


def test_a_corpus_absent_from_the_pack_is_named(tmp_path):
    from senbonzakura import bundled
    blob = tmp_path / corpora.CORPORA_BLOB
    blob.write_bytes(bundled.pack(json.dumps(
        {"schema": "senbonzakura-corpora/1", "corpora": {}}).encode("utf-8")))
    with pytest.raises(CorpusError, match="different version"):
        corpora.load("advbench", root=tmp_path)


# ── reaching them the way a user does ────────────────────────────────────────────
@needs_pack
def test_a_bundled_corpus_resolves_by_name():
    assert len(dataset.resolve("advbench", what="prompt set")) == 520


@needs_pack
def test_a_limit_applies_through_the_normal_path():
    assert len(dataset.resolve("advbench", limit=64, what="prompt set")) == 64


@needs_pack
def test_a_local_directory_cannot_shadow_a_bundled_name(tmp_path, monkeypatch):
    """The same guard `default` has: a stray directory must not silently become the corpus a
    published number came from.
    """
    (tmp_path / "advbench").mkdir()
    monkeypatch.chdir(tmp_path)
    assert len(dataset.resolve("advbench", what="prompt set")) == 520


def test_the_ambiguous_name_is_refused_through_resolve_too():
    with pytest.raises(CorpusError):
        dataset.resolve("xstest", what="prompt set")


# ── the prerequisite is named before the first download ──────────────────────────
#
# `doctor` sends a stranger to `tools/build_corpora.py` by name when the corpora are missing, so
# it gets run on machines that were never set up for it. On one without the GitHub CLI it died on
# a raw `FileNotFoundError: 'gh'` out of subprocess, which names the missing program and nothing
# a person can act on. Found by running it on the CPU box.
def _build_corpora_module():
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parent.parent / "tools" / "build_corpora.py"
    spec = importlib.util.spec_from_file_location("_build_corpora_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_missing_github_cli_is_refused_with_something_to_do(monkeypatch):
    mod = _build_corpora_module()
    monkeypatch.setattr(mod.shutil, "which", lambda _name: None)
    with pytest.raises(mod.BuildError) as e:
        mod.preflight()
    msg = str(e.value)
    assert "gh" in msg and "cli.github.com" in msg, msg
    assert "auth login" in msg, "the message names the program but not how to make it usable"


def test_the_preflight_passes_when_the_cli_is_present(monkeypatch):
    mod = _build_corpora_module()
    monkeypatch.setattr(mod.shutil, "which", lambda _name: "/usr/bin/gh")
    mod.preflight()


def test_fetch_itself_also_refuses_rather_than_raising_oserror(monkeypatch):
    """Belt and braces: anything calling fetch() directly gets the sentence, not the OSError."""
    mod = _build_corpora_module()

    def _no_gh(*_a, **_k):
        raise FileNotFoundError(2, "No such file or directory", "gh")

    monkeypatch.setattr(mod.subprocess, "run", _no_gh)
    with pytest.raises(mod.BuildError) as e:
        mod.fetch("owner/repo", "abc123", "some.csv")
    assert "gh" in str(e.value)


def test_the_runner_refuses_and_says_why_end_to_end(monkeypatch, capsys):
    """Behavioural, so that deleting the preflight() call from main() fails this rather than
    leaving three green tests that only ever exercised the function directly.
    """
    mod = _build_corpora_module()
    monkeypatch.setattr(mod.shutil, "which", lambda _name: None)

    def _should_not_run(*_a, **_k):
        raise AssertionError("a download was attempted after the prerequisite check failed")

    monkeypatch.setattr(mod.subprocess, "run", _should_not_run)
    assert mod.main([]) == 1
    assert "cli.github.com" in capsys.readouterr().err


def test_the_attribution_notice_prints_once_per_process():
    """The licence asks for the notice, not for it once per row."""
    from senbonzakura import corpora

    corpora._reset_notice_for_tests()
    said = []
    corpora.notice("advbench", log=said.append)
    first = len(said)
    corpora.notice("xstest-safe", log=said.append)
    assert first > 0
    assert len(said) == first, "the notice repeated within one process"


def test_a_corrupt_pack_says_which_tool_rebuilds_it(tmp_path, monkeypatch):
    """A pack that decrypts to rubbish would otherwise fit a direction on noise."""
    import pytest

    from senbonzakura import bundled, corpora

    blob = tmp_path / "corpora.bin"
    blob.write_bytes(bundled.pack(b"this is not json"))
    monkeypatch.setattr(bundled, "data_path", lambda: tmp_path / "default-track.bin")
    with pytest.raises(corpora.CorpusError, match=r"build_corpora\.py"):
        corpora.load("advbench", root=tmp_path)
