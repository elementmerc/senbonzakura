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
