# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""A wheel that carries a Linux executable must not claim to run anywhere.

`pip install senbonzakura` gets a `py3-none-any` wheel that cannot convert or quantise, because
the binaries are not in it. Per-platform wheels fix that, and introduce a worse failure if the tag
and the contents are allowed to disagree: a universal wheel holding a Linux binary installs
happily on macOS and Windows and dies at first use.

`setup.py` makes the tag follow the contents, so that combination cannot be built. These tests
hold that down, including the trap that defeated it on the first attempt: `build/lib/` survives
between builds and setuptools copies from it.
"""
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_setup(monkeypatch, root):
    """Import `setup.py` without letting it run `setup()`, with its paths pointed at `root`."""
    import setuptools
    monkeypatch.setattr(setuptools, "setup", lambda **_kw: None)
    spec = importlib.util.spec_from_file_location("_setup_under_test", ROOT / "setup.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_setup_under_test"] = mod
    spec.loader.exec_module(mod)
    mod.VENDOR_BIN = root / "src" / "senbonzakura" / "vendor" / "bin"
    mod.STAGED_BIN = root / "build" / "lib" / "senbonzakura" / "vendor" / "bin"
    return mod


def place(root, where, platform):
    d = root / where / platform
    d.mkdir(parents=True, exist_ok=True)
    (d / "llama-quantize").write_text("#!/bin/sh\n", encoding="utf-8")


# ── the tag follows the contents ─────────────────────────────────────────────────
def test_nothing_vendored_means_a_universal_wheel(monkeypatch, tmp_path):
    mod = load_setup(monkeypatch, tmp_path)
    assert mod.vendored_platforms() == []


def test_a_vendored_platform_is_found(monkeypatch, tmp_path):
    place(tmp_path, "src/senbonzakura/vendor/bin", "linux-x86_64")
    mod = load_setup(monkeypatch, tmp_path)
    assert mod.vendored_platforms() == ["linux-x86_64"]


def test_a_directory_without_an_executable_does_not_count(monkeypatch, tmp_path):
    """An empty platform directory is left behind by a failed vendor run. It is not a payload,
    and treating it as one would tag a wheel for a platform whose binaries it does not have.
    """
    (tmp_path / "src/senbonzakura/vendor/bin/macos-arm64").mkdir(parents=True)
    mod = load_setup(monkeypatch, tmp_path)
    assert mod.vendored_platforms() == []


# ── the two ways it can go wrong ─────────────────────────────────────────────────
def test_two_vendored_platforms_are_refused(monkeypatch, tmp_path):
    """One wheel, one tag. Building with several vendored would pick one and ship the rest as
    dead weight inside a wheel that does not admit to carrying them.
    """
    for p in ("linux-x86_64", "macos-arm64"):
        place(tmp_path, "src/senbonzakura/vendor/bin", p)
    mod = load_setup(monkeypatch, tmp_path)
    assert len(mod.vendored_platforms()) == 2


def test_a_stale_build_tree_is_detected(monkeypatch, tmp_path):
    """THE TRAP, and it is not hypothetical: it happened on the first attempt to verify the empty
    case. Source tree has no binaries, so the tag is `any`; `build/lib/` still has them, so they
    go into the wheel anyway. That is exactly the artefact setup.py exists to make unbuildable.
    """
    place(tmp_path, "build/lib/senbonzakura/vendor/bin", "linux-x86_64")
    mod = load_setup(monkeypatch, tmp_path)
    assert mod.vendored_platforms() == []
    assert mod._staged_platforms() == ["linux-x86_64"]
    stale = [x for x in mod._staged_platforms() if x not in mod.vendored_platforms()]
    assert stale == ["linux-x86_64"], "the mismatch that produces a mislabelled wheel"


def test_a_matching_build_tree_is_not_stale(monkeypatch, tmp_path):
    """The normal case: the staged tree agrees with the source tree, and nothing is wrong."""
    place(tmp_path, "src/senbonzakura/vendor/bin", "linux-x86_64")
    place(tmp_path, "build/lib/senbonzakura/vendor/bin", "linux-x86_64")
    mod = load_setup(monkeypatch, tmp_path)
    assert [x for x in mod._staged_platforms() if x not in mod.vendored_platforms()] == []


# ── the invariant, on the real tree ──────────────────────────────────────────────
def test_the_shipped_wheel_never_disagrees_with_itself():
    """Whatever this checkout currently is, the wheel it builds must not lie.

    A universal tag and a binary payload is the one combination that installs on a machine it
    cannot run on. Asserted against a real build rather than reasoned about, because the last two
    times this was reasoned about the reasoning was wrong.
    """
    import glob
    import subprocess
    import zipfile
    out = ROOT / "dist-invariant"
    subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(out)],
                   cwd=ROOT, capture_output=True, text=True, timeout=900, check=False)
    wheels = glob.glob(str(out / "*.whl"))
    if not wheels:
        pytest.skip("the wheel would not build in this environment")
    name = pathlib.Path(wheels[-1]).name
    names = zipfile.ZipFile(wheels[-1]).namelist()
    has_binary = any("llama-quantize" in n or "llama-imatrix" in n for n in names)
    universal = name.endswith("-py3-none-any.whl")
    assert not (universal and has_binary), (
        f"{name} says it runs anywhere and carries a platform executable")
    # And the other direction, which is a different claim: a platform tag with nothing in it to
    # justify one would refuse installation everywhere else for no reason at all.
    assert not (not universal and not has_binary), (
        f"{name} is tagged for a platform and carries no platform payload")
    for w in wheels:
        pathlib.Path(w).unlink()
    out.rmdir()


# ── the header tool must not silently disable an encoding declaration ────────────
def test_a_licence_header_does_not_push_an_encoding_declaration_off_line_two(tmp_path):
    """PEP 263 honours `# -*- coding: ... -*-` only on line 1 or 2.

    Inserting the licence between the shebang and the declaration moved it to line 3 and turned
    it off silently. Nothing in this repository carries one today, so it was latent rather than
    live, and a latent corruption in a tool that rewrites every file in the tree is worth closing.
    """
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "tools"))
    import add_license_headers as alh

    f = tmp_path / "legacy.py"
    f.write_text('#!/usr/bin/env python3\n# -*- coding: latin-1 -*-\n"""D."""\n', encoding="utf-8")
    alh.apply(f)
    lines = f.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("#!")
    assert "coding" in lines[1], "the encoding declaration must stay on line 1 or 2"
    assert lines[2].startswith("# SPDX")


def test_a_file_with_only_an_encoding_declaration_keeps_it_first(tmp_path):
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "tools"))
    import add_license_headers as alh

    f = tmp_path / "enc.py"
    f.write_text("# -*- coding: utf-8 -*-\nx = 1\n", encoding="utf-8")
    alh.apply(f)
    assert f.read_text(encoding="utf-8").splitlines()[0].startswith("# -*- coding")


# ── the standalone wheel check, and its negative control ────────────────────────────
def _wheel_check():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "check_wheel", ROOT / "tools" / "check_wheel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_wheel(tmp_path, name, *, payload=False, metadata_tag=None):
    """A minimal wheel-shaped zip, so the check can be exercised without a real build."""
    import zipfile
    tag = metadata_tag or "-".join(name[: -len(".whl")].split("-")[-3:])
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("pkg/__init__.py", "")
        z.writestr("pkg-0.0.0.dist-info/WHEEL", f"Wheel-Version: 1.0\nTag: {tag}\n")
        if payload:
            z.writestr("pkg/vendor/bin/llama-quantize", "not really a binary")
    return path


def test_an_honest_universal_wheel_passes(tmp_path):
    cw = _wheel_check()
    w = _fake_wheel(tmp_path, "pkg-0.0.0-py3-none-any.whl")
    assert cw.problems(cw.inspect(w)) == []
    assert cw.main([str(w)]) == 0


def test_an_honest_platform_wheel_passes(tmp_path):
    cw = _wheel_check()
    w = _fake_wheel(tmp_path, "pkg-0.0.0-py3-none-linux_x86_64.whl", payload=True)
    assert cw.problems(cw.inspect(w)) == []


def test_a_universal_tag_with_a_binary_is_the_combination_that_breaks_for_strangers(tmp_path):
    """It installs on a Mac, a Windows box and an ARM server, and fails on all three."""
    cw = _wheel_check()
    w = _fake_wheel(tmp_path, "pkg-0.0.0-py3-none-any.whl", payload=True)
    found = cw.problems(cw.inspect(w))
    assert any("runs anywhere" in p for p in found)
    assert cw.main([str(w)]) == 1


def test_a_platform_tag_with_nothing_to_justify_it_is_also_wrong(tmp_path):
    """The quieter waste: refusing installation everywhere else for no reason."""
    cw = _wheel_check()
    w = _fake_wheel(tmp_path, "pkg-0.0.0-py3-none-linux_x86_64.whl")
    assert any("nothing" in p or "no platform payload" in p
               for p in cw.problems(cw.inspect(w)))


def test_a_renamed_wheel_is_caught_by_its_own_metadata(tmp_path):
    """Renaming a file is the easiest way to make a wheel lie, and pip believes the metadata."""
    cw = _wheel_check()
    w = _fake_wheel(tmp_path, "pkg-0.0.0-py3-none-any.whl",
                    payload=True, metadata_tag="py3-none-linux_x86_64")
    found = cw.problems(cw.inspect(w))
    assert any("filename says" in p and "metadata says" in p for p in found)


def test_a_wheel_with_no_metadata_is_refused(tmp_path):
    import zipfile
    cw = _wheel_check()
    path = tmp_path / "pkg-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("pkg/__init__.py", "")
    assert any("no WHEEL metadata" in p for p in cw.problems(cw.inspect(path)))


def test_expect_failure_inverts_the_verdict_both_ways(tmp_path):
    """THE CONTROL. CI runs this against a wheel built to be wrong.

    A gate only ever seen agreeing has not been shown to work, so the mode that requires a
    rejection is itself tested in both directions: it must succeed on a bad wheel and fail on a
    good one.
    """
    cw = _wheel_check()
    bad = _fake_wheel(tmp_path, "pkg-0.0.0-py3-none-any.whl", payload=True)
    good = _fake_wheel(tmp_path, "ok-0.0.0-py3-none-any.whl")
    assert cw.main([str(bad), "--expect-failure"]) == 0
    assert cw.main([str(good), "--expect-failure"]) == 1


def test_a_missing_or_unreadable_file_exits_two_not_one(tmp_path):
    """Distinguished so CI can tell 'the wheel is wrong' from 'the check could not run'."""
    cw = _wheel_check()
    assert cw.main([str(tmp_path / "nope.whl")]) == 2
    junk = tmp_path / "junk-0.0.0-py3-none-any.whl"
    junk.write_text("not a zip")
    assert cw.main([str(junk)]) == 2


def test_a_name_that_is_not_a_wheel_is_refused(tmp_path):
    cw = _wheel_check()
    with pytest.raises(ValueError, match="not a wheel filename"):
        cw._filename_tag("thing.whl")


# ── the tag PyPI refuses, caught before the release rather than after it (S7) ─────────
#
# This tree builds `py3-none-linux_x86_64` once tools/vendor_llama.py has run, which is the
# correct tag for the contents and is NOT one PyPI accepts. `check_wheel` asked only whether the
# tag matched the contents, and `twine check` validates metadata renderability and says nothing
# about platform tags, so both gates passed it.
#
# The order of events is what makes it expensive. publish.yml downloads artefacts from a GitHub
# Release that already exists, checks them, and uploads. The refusal lands at the upload, by which
# time the Release is public, the tag is pushed, and the version number is spent: PyPI will not
# take that version again even once the wheel is fixed.

_spec = importlib.util.spec_from_file_location(
    "check_wheel", ROOT / "tools" / "check_wheel.py")
check_wheel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_wheel)


def _info(tag):
    return {"filename_tag": tag, "metadata_tags": [tag]}


def test_the_tag_this_tree_builds_with_binaries_is_refused_for_pypi():
    """THE DEFECT. `linux_x86_64` is what setup.py produces when the binaries are vendored."""
    bad = check_wheel.unacceptable_to_pypi(_info("py3-none-linux_x86_64"))
    assert bad, "the tag PyPI actually rejects has to be rejected here first"
    assert "after the GitHub Release is already published" in bad[0], (
        "the message has to say WHEN the failure lands, because that is what makes it expensive")
    assert "auditwheel" in bad[0] and "attach this one to the Release" in bad[0], (
        "and name both ways out")


@pytest.mark.parametrize("tag", [
    "py3-none-any",
    "py3-none-manylinux_2_28_x86_64",
    "py3-none-musllinux_1_2_x86_64",
    "py3-none-macosx_11_0_arm64",
    "py3-none-win_amd64",
])
def test_every_tag_pypi_does_accept_passes(tag):
    """A gate that refuses everything is as useless as one that refuses nothing."""
    assert check_wheel.unacceptable_to_pypi(_info(tag)) == []


def test_a_disagreement_between_the_two_tags_is_caught_on_either():
    """A wheel states its tag twice and the two can disagree; both are read."""
    assert check_wheel.unacceptable_to_pypi(
        {"filename_tag": "py3-none-any", "metadata_tags": ["py3-none-linux_x86_64"]})
    assert check_wheel.unacceptable_to_pypi(
        {"filename_tag": "py3-none-linux_x86_64", "metadata_tags": ["py3-none-any"]})


def test_a_wheel_with_no_metadata_tag_is_still_read_from_its_filename():
    assert check_wheel.unacceptable_to_pypi(
        {"filename_tag": "py3-none-linux_x86_64", "metadata_tags": None})


# ─────────────────────────────────────────────────────────────────────────────────────
# The build machine's filesystem, which shipped to every user for two releases.
#
# The 2026-09-10 panel unpacked `src/senbonzakura/data/default-track.bin` from the
# published wheel and read this out of the track manifest inside it:
#
#     "labels":  "/home/heph-agent/track2-enriched-backup/contrast/axis-labels-both.tsv"
#     "harmful": "/tmp/senbon-rebuild/harmful.txt"
#
# That unpacks into every user's ~/.cache/senbonzakura/bundled-track/. Baseline 13
# forbids private paths in shipped artefacts. `74e571f` fixed the same class of defect
# one level up and this survived, because the blob predates it and was never repacked,
# which is the argument for a gate rather than a memory.
# ─────────────────────────────────────────────────────────────────────────────────────

def _wheel_with(tmp_path, name, body):
    import zipfile
    w = tmp_path / "senbonzakura-9.9.9-py3-none-any.whl"
    with zipfile.ZipFile(w, "w") as z:
        z.writestr(name, body)
    return w


def test_a_build_machine_path_in_a_shipped_blob_is_refused(tmp_path):
    w = _wheel_with(tmp_path, "senbonzakura/data/track.json",
                    '{"sources": {"harmful": "/home/heph-agent/rebuild/harmful.txt"}}')
    problems = check_wheel.leaks_a_build_path(w)
    assert problems, "the exact string read out of the published wheel was not flagged"
    assert "/home/heph-agent" in problems[0]
    assert "pack_track" in problems[0], "the reader needs the remedy, not just the complaint"


def test_a_scrubbed_manifest_passes(tmp_path):
    w = _wheel_with(tmp_path, "senbonzakura/data/track.json",
                    '{"sources": {"harmful": "harmful.txt"}}')
    assert check_wheel.leaks_a_build_path(w) == []


def test_a_tmp_path_counts_too(tmp_path):
    w = _wheel_with(tmp_path, "senbonzakura/data/track.json",
                    '{"sources": {"harmful": "/tmp/senbon-rebuild/harmful.txt"}}')
    assert check_wheel.leaks_a_build_path(w)


def test_source_files_are_not_scanned(tmp_path):
    """Docstrings and comments mention /tmp legitimately, and a gate that flagged those would be
    switched off within a week. Only the shipped data blobs are read.
    """
    w = _wheel_with(tmp_path, "senbonzakura/bundled.py",
                    'CACHE = "/home/someone/.cache"  # an example in a comment\n')
    assert check_wheel.leaks_a_build_path(w) == []


def test_the_real_shipped_blob_carries_no_build_machine_path():
    """The artefact itself, not a fixture of it.

    A fixture is a claim about the world; this reads the blob that will actually ship.
    """
    import json
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from senbonzakura import bundled
    if not bundled.is_available():
        pytest.skip("no bundled track in this checkout")
    track = pathlib.Path(bundled.ensure()) / "track.json"
    doc = json.loads(track.read_text(encoding="utf-8"))
    for key, value in (doc.get("sources") or {}).items():
        assert not str(value).startswith(("/home/", "/tmp/", "/root/", "/Users/")), (
            f"the shipped track manifest still carries the build machine's path for {key}: {value}")
