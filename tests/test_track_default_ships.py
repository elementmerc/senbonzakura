# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`--track default` works for the people who install this, or the release does not go out.

WHAT WAS WRONG, measured 2026-09-10

A wheel built from a plain clone carries neither `senbonzakura/data/default-track.bin` nor
`senbonzakura/data/corpora.bin`, because both are generated artefacts kept out of git on purpose.
It installs, imports and answers `--help` happily, and then `--track default` fails for everyone
who installed it. The documentation offers that flag as the shortest path to a first run and the
tool's own not-found hint recommends it, so the failure lands on exactly the person least equipped
to work out why.

This is a defect waiting for the next release rather than one live on the index: the published
0.3.0 predates the bundled track entirely, with seven modules, no `bundled` module and no
`--track default` to fail. That was checked after the first version of this file asserted the
opposite, which is the same move the file is about.

The mechanism was already understood and already written down. RELEASING.md says the two blobs are
generated artefacts kept out of git on purpose, that a wheel built from a plain clone "installs
happily, imports happily, answers `--help` happily, and then fails `--track default` for every
person who installs it", and that "`--release` is the part that is easy to skip and expensive to
skip".

That last sentence is the whole defect. The gate existed and worked; the only thing standing
between a hollow wheel and PyPI was somebody remembering to type a flag.

THE TWO HALVES FIXED HERE

A flag can be forgotten and a version cannot, so the artefact is asked rather than the operator:
a wheel naming this project at a release version faces the release checks whether or not anyone
typed `--release`.

And when an install does turn out to be hollow, the message it gives has to be the right one. A
source checkout and an installed wheel need opposite advice, and both used to get one sentence
that pointed a wheel user at a tool their install had never contained.
"""
import zipfile
from pathlib import Path

import pytest

import tools.check_wheel as cw
from senbonzakura import bundled


def _wheel(tmp_path, name, entries=()):
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as z:
        stem = name.split("-py3")[0]
        z.writestr(f"{stem}.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
        z.writestr(f"{stem}.dist-info/METADATA", "Metadata-Version: 2.1\n")
        for e in entries:
            z.writestr(e, "x")
    return p


# ── the flag that was easy to forget ─────────────────────────────────────────────────

def test_our_wheel_at_a_release_version_is_a_release_artefact():
    assert cw.is_release_artefact(Path("senbonzakura-0.4.0-py3-none-any.whl"))
    assert cw.is_release_artefact(Path("senbonzakura-1.0-py3-none-any.whl"))


@pytest.mark.parametrize("name", [
    "senbonzakura-0.4.0.dev0-py3-none-any.whl",     # what a clean checkout builds
    "senbonzakura-0.4.0rc1-py3-none-any.whl",       # a pre-release is allowed to be thin
    "senbonzakura-0.4.0.dev1+local-py3-none-any.whl",
])
def test_a_pre_release_of_ours_is_not_forced_into_release_mode(name):
    assert not cw.is_release_artefact(Path(name))


@pytest.mark.parametrize("name", [
    "pkg-0.0.0-py3-none-any.whl",        # the synthetic wheels this tool is pointed at
    "ok-1.2.3-py3-none-any.whl",
    "senbonzakura_extras-1.0-py3-none-any.whl",
])
def test_somebody_else_s_wheel_is_left_alone(name):
    """The check is about OUR release artefact. Fixtures carry no corpora and are not supposed to."""
    assert not cw.is_release_artefact(Path(name))


def test_a_hollow_release_wheel_is_refused_with_no_flag_at_all(tmp_path, capsys):
    """THE DEFECT. This is the shape of the wheel that reached PyPI as 0.3.0."""
    w = _wheel(tmp_path, "senbonzakura-0.4.0-py3-none-any.whl")
    assert cw.main([str(w)]) == 1
    out = capsys.readouterr().out
    assert "release checks  ON" in out, "the reader must be told why it was stricter than asked"
    assert "default-track.bin" in out
    assert "corpora.bin" in out


def test_the_same_wheel_at_a_dev_version_passes(tmp_path):
    """A wheel built from a clean checkout is fine for CI and must not be failed here."""
    assert cw.main([str(_wheel(tmp_path, "senbonzakura-0.4.0.dev0-py3-none-any.whl"))]) == 0


# ── the message an install with no track gives ───────────────────────────────────────

def test_a_checkout_is_told_to_build_the_blob(monkeypatch, tmp_path):
    monkeypatch.setattr(bundled, "is_available", lambda: False)
    monkeypatch.setattr(bundled, "_cache_is_current", lambda t: False)
    monkeypatch.setattr(bundled, "running_from_a_checkout", lambda: True)
    monkeypatch.setattr(bundled, "cache_dir", lambda: tmp_path / "cache")
    with pytest.raises(bundled.BundledTrackError) as e:
        bundled.ensure(log=lambda *a, **k: None)
    assert "tools/pack_track.py" in str(e.value)
    assert "report" not in str(e.value).lower(), "a checkout is not a bug report"


def test_an_install_is_told_it_is_a_packaging_fault_and_where_to_say_so(monkeypatch, tmp_path):
    """THE ONE THAT MATTERED. The old sentence sent a wheel user to a file they never had.

    They cannot run `tools/pack_track.py`, because it is not in their install, and they cannot
    build their own track, because that needs a corpus of harmful prompts they have no way to get.
    Being told to do either is worse than being told nothing: it reads as their mistake.
    """
    monkeypatch.setattr(bundled, "is_available", lambda: False)
    monkeypatch.setattr(bundled, "_cache_is_current", lambda t: False)
    monkeypatch.setattr(bundled, "running_from_a_checkout", lambda: False)
    monkeypatch.setattr(bundled, "cache_dir", lambda: tmp_path / "cache")
    with pytest.raises(bundled.BundledTrackError) as e:
        bundled.ensure(log=lambda *a, **k: None)
    msg = str(e.value)
    assert "tools/pack_track.py" not in msg, "that file is not in their install"
    assert "fault in the package rather than anything you did" in msg
    assert bundled.ISSUES in msg, "told it is a bug and given nowhere to report it is still stuck"
    assert bundled._installed_version() in msg, "a report without a version cannot be acted on"


def test_the_checkout_test_asks_for_the_packer_rather_than_for_git():
    """A `git archive` extract and an sdist are both source trees with no repository.

    What decides the advice is whether the command in the message exists to be run, so that is
    what is asked. This repository has the packer, so it reads as a checkout.
    """
    assert bundled.running_from_a_checkout()


# ── the hint the tool gives when a track is missing ──────────────────────────────────

def test_the_hint_offers_the_bundled_track_only_when_it_is_there(monkeypatch, tmp_path):
    """It said "the one bundled in this install" without asking this install.

    On the wheel published as 0.3.0 there is no bundled track, so the tool recommended a flag it
    could not honour and the user's next command failed for a second reason. An unverified hint is
    a guess in the imperative mood.
    """
    from argparse import Namespace

    from senbonzakura import cli

    args = Namespace(track=str(tmp_path / "nope"))

    monkeypatch.setattr(bundled, "is_available", lambda: True)
    with pytest.raises(SystemExit) as e:
        cli._preflight_datasets(args)
    assert "--track default" in str(e.value)

    monkeypatch.setattr(bundled, "is_available", lambda: False)
    with pytest.raises(SystemExit) as e:
        cli._preflight_datasets(args)
    msg = str(e.value)
    assert "will not help here either" in msg
    assert "senbonzakura track --help" in msg


# ── the extraction's own cleanup paths, which nothing exercised ──────────────────────

def test_a_stamp_that_cannot_be_read_is_treated_as_no_cache(monkeypatch, tmp_path):
    """An unreadable stamp means the question cannot be answered, so the cache is rebuilt.

    Trusting a cache whose stamp could not be read is how half a track gets used as a whole one.
    """
    target = tmp_path / "cache"
    target.mkdir()
    stamp = target / bundled.STAMP_NAME
    stamp.write_text("whatever", encoding="utf-8")

    def boom(*a, **k):
        raise OSError("stamp unreadable")

    monkeypatch.setattr(Path, "read_text", boom)
    assert bundled._cache_is_current(target) is False


def test_an_interrupted_extraction_leaves_no_half_track_behind(monkeypatch, tmp_path):
    """The leftover `.unpacking` directory is cleared before a retry, not extracted on top of.

    Exercises the branch where a previous run died mid-extraction: the stamp is written last, so
    the half-written cache fails its check and is redone rather than trusted.
    """
    cache = tmp_path / "cache"
    stale = cache.with_name(cache.name + ".unpacking")
    stale.mkdir(parents=True)
    (stale / "leftover").write_text("from a run that died", encoding="utf-8")

    monkeypatch.setattr(bundled, "cache_dir", lambda: cache)
    monkeypatch.setattr(bundled, "_reset_notice_for_tests", lambda: None)
    bundled._state["notified"] = True
    got = bundled.ensure(log=lambda *a, **k: None)
    assert got == cache
    assert not stale.exists(), "the interrupted extraction was left on disk"
    assert (cache / bundled.STAMP_NAME).is_file(), "the stamp is written last, and it is missing"
    assert not (cache / "leftover").exists(), "the half-written tree was extracted on top of"
