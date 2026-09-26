# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The MIT notice is a precondition of placing the vendored binaries, not a courtesy.

WHAT WENT WRONG. `extract()` copied `vendor/src/LICENSE` beside the extracted binaries only if
that file already existed, and warned otherwise. But the binaries were vendored BEFORE the source
tree, and the source step is the only one that fetches the notice; both directories are ignored,
so every clean checkout had neither, took the warning branch, and produced roughly 35 MB of
MIT-licensed ggml and llama.cpp object code with no notice beside it. The Dockerfile copies that
directory into the image and `distribute.yml` pushes the image to ghcr on every push to `dev`.
MIT asks for the notice in "all copies or substantial portions".

The comment above the branch said "Placed here rather than by hand so a re-vendor cannot drop it
again" and THIRD-PARTY-NOTICES.md said so publicly. Both were false, on every run. Found by the
review panel on 2026-09-25.

These tests pin the three halves of the fix: the placement raises rather than warns, the fetch
order puts the source tree first, and `vendor_binaries` refuses before it spends a download.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "vendor_llama", REPO / "tools" / "packaging" / "vendor_llama.py")
vendor = importlib.util.module_from_spec(_SPEC)
sys.modules["vendor_llama"] = vendor
_SPEC.loader.exec_module(vendor)


def _archive(tmp_path, names):
    """A tar.gz shaped like a llama.cpp release asset: the wanted binaries under build/bin/."""
    import io
    import tarfile

    p = tmp_path / "asset.tar.gz"
    with tarfile.open(p, "w:gz") as tf:
        for name in names:
            data = b"#!/bin/sh\necho usage\n"
            info = tarfile.TarInfo(f"build/bin/{name}")
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
    return p


def _vendor_tree(tmp_path, *, with_licence: bool):
    """A `vendor/` directory laid out the way the real one is: bin/<platform>/ beside src/."""
    out_dir = tmp_path / "vendor" / "bin" / "linux-x86_64"
    src = tmp_path / "vendor" / "src"
    src.mkdir(parents=True)
    if with_licence:
        (src / "LICENSE").write_text("MIT License\n\nCopyright (c) the ggml authors\n",
                                     encoding="utf-8")
    return out_dir


class TestThePlacementRefusesRatherThanWarns:
    def test_extraction_without_the_notice_raises(self, tmp_path):
        out_dir = _vendor_tree(tmp_path, with_licence=False)
        archive = _archive(tmp_path, vendor.WANT_BINS)
        with pytest.raises(vendor.VendorFetchError) as e:
            vendor.extract(archive, out_dir, log=lambda *a: None)
        assert "licence text" in str(e.value)
        assert "--skip-script" in str(e.value)

    def test_nothing_is_written_at_all(self, tmp_path):
        """A refusal that still leaves the object code on disk is a warning with extra steps.

        The Dockerfile copies `vendor/bin` into the image wholesale, so 35 MB left behind by a
        failed extraction would ship exactly as an unlicensed copy.
        """
        out_dir = _vendor_tree(tmp_path, with_licence=False)
        archive = _archive(tmp_path, vendor.WANT_BINS)
        with pytest.raises(vendor.VendorFetchError):
            vendor.extract(archive, out_dir, log=lambda *a: None)
        assert not out_dir.exists()

    def test_with_the_notice_present_it_is_placed_beside_them(self, tmp_path):
        out_dir = _vendor_tree(tmp_path, with_licence=True)
        archive = _archive(tmp_path, vendor.WANT_BINS)
        written = vendor.extract(archive, out_dir, log=lambda *a: None)
        assert "LICENSE" in written
        assert (out_dir / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")


class TestTheFetchOrder:
    def test_vendor_binaries_pre_flights_the_notice_before_downloading(self, monkeypatch, tmp_path):
        """Refused before a byte moves. Finding out after six platform archives is a bad way."""
        monkeypatch.setattr(vendor.vendored, "VENDOR_SRC", tmp_path / "nowhere")

        def boom(*a, **k):
            raise AssertionError("a download was started before the notice was checked")

        monkeypatch.setattr(vendor, "_download", boom)
        manifest = {"pins": {"llama.cpp": {"repo": "x/y", "tag": "b1", "assets": {
            "linux-x86_64": {"file": "a.tar.gz", "size": 1}}}}}
        with pytest.raises(vendor.VendorFetchError) as e:
            vendor.vendor_binaries(manifest, ["linux-x86_64"], log=lambda *a: None)
        assert "MIT notice" in str(e.value)

    def test_a_dry_run_is_not_blocked_by_it(self, monkeypatch, tmp_path):
        """`--dry-run` prints URLs and places nothing, so it has nothing to be licensed."""
        monkeypatch.setattr(vendor.vendored, "VENDOR_SRC", tmp_path / "nowhere")
        manifest = {"pins": {"llama.cpp": {"repo": "x/y", "tag": "b1", "assets": {
            "linux-x86_64": {"file": "a.tar.gz", "size": 1}}}}}
        assert vendor.vendor_binaries(manifest, ["linux-x86_64"], dry_run=True,
                                      log=lambda *a: None) == {}

    def test_the_source_tree_is_fetched_first(self):
        """The order in `main` is load-bearing, and reading it is how this defect is prevented.

        The source step is the ONLY one that fetches llama.cpp's LICENSE. Run after the binaries,
        as it was until 2026-09-25, the notice can never be present on a clean checkout.
        """
        text = (REPO / "tools" / "packaging" / "vendor_llama.py").read_text(encoding="utf-8")
        body = text.split("def main(", 1)[1]
        assert body.index("vendor_conversion(") < body.index("vendor_binaries(")


class TestTheArtefactIsAsserted:
    def test_the_distribute_workflow_reads_the_image_not_the_tool(self):
        """A gate that reads the tool's intent is the shape that let this escape.

        `vendor_llama.py` carried a comment saying a re-vendor could not drop the notice while it
        was dropping it on every run. So the workflow now looks inside the built image, before
        the push, for a LICENSE in every platform directory.
        """
        text = (REPO / ".github" / "workflows" / "distribute.yml").read_text(encoding="utf-8")
        gate = text.index("MIT notice ships beside the object code")
        push = text.rindex("push: true")
        assert gate < push, "the notice check runs after the push, so it cannot gate it"
        assert "vendor/bin" in text[gate:push]

    def test_the_checked_in_tree_carries_the_notice_when_it_carries_binaries(self):
        """Skipped where nothing is vendored, which is every clean checkout and most of CI."""
        root = REPO / "src" / "senbonzakura" / "vendor" / "bin"
        platforms = [d for d in root.glob("*") if d.is_dir()] if root.is_dir() else []
        if not platforms:
            pytest.skip("nothing vendored here, so there is no artefact to check")
        for d in platforms:
            assert (d / "LICENSE").is_file(), f"{d} carries object code and no MIT notice"
