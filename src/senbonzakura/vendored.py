# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Find a vendored third-party executable, or say exactly why there is not one.

WHAT IS VENDORED AND WHAT IS NOT, AND WHY THEY DIFFER

Two kinds of artefact are pinned in `vendor/pins.json`, and they are handled differently on
purpose:

  * `convert_hf_to_gguf.py` is **committed to the repository.** It is source, so vendoring it the
    way you vendor source means a reader can see in a diff exactly what changed when the pin moved.
    That is the whole value of vendoring a script rather than fetching one.
  * The **binaries are not committed.** Six platform archives is roughly eighty megabytes of
    opaque blob per pin bump, in a public repository, reviewable by nobody. They are downloaded at
    wheel-build time by `tools/vendor_llama.py` and travel inside the wheel instead.

So a git checkout has the script and no binaries, and an installed wheel has both. That asymmetry
is the reason this module exists rather than a hardcoded path: the same code has to work in a
checkout, in a wheel, and on a platform nothing was built for.

THE RESOLUTION ORDER, AND WHY THE SYSTEM COPY IS NOT FIRST

1. The binary vendored into this install, if one is present for this platform.
2. A copy on PATH.
3. A loud failure naming both routes.

Vendored first, because a comparison has to be able to say which build produced it, and silently
preferring whatever a machine happens to have installed is how two arms of one experiment get
different tools. The PATH fallback exists for the platforms nothing was built for, and it warns
rather than passing quietly, because at that point the version is whatever that machine has.
"""
from __future__ import annotations

import os
import platform
import shutil
import stat
import sys
from pathlib import Path

from .vendoring import VendorError, load_manifest

#: Where `tools/vendor_llama.py` places what it extracts, and where the wheel carries it.
VENDOR_BIN = Path(__file__).resolve().parent / "vendor" / "bin"

#: Where the vendored pure-Python scripts live. These ARE committed.
VENDOR_SRC = Path(__file__).resolve().parent / "vendor" / "src"


def platform_key():
    """The `os-arch` key used in the manifest and on disk, or None on something unrecognised.

    Normalised rather than passed through: `platform.machine()` says `x86_64` on Linux, `AMD64` on
    Windows and `arm64` or `aarch64` depending on the box, and four spellings of two machines is
    how a lookup misses a binary that is sitting right there.
    """
    arch = {
        "x86_64": "x86_64", "amd64": "x86_64", "AMD64": "x86_64",
        "aarch64": "aarch64", "arm64": "aarch64",
    }.get(platform.machine())
    if arch is None:
        return None
    if sys.platform.startswith("linux"):
        return f"linux-{arch}"
    if sys.platform == "darwin":
        # macOS keys keep Apple's own spelling, because that is what the upstream archives use
        # and a translation layer between two naming schemes is a place to be wrong twice.
        return "macos-arm64" if arch == "aarch64" else "macos-x86_64"
    if sys.platform.startswith("win"):
        return "windows-arm64" if arch == "aarch64" else "windows-x86_64"
    return None


def _executable(p):
    return p.is_file() and os.access(p, os.X_OK)


def find_binary(name, *, key=None, search_path=True, log=None):
    """Resolve an executable to (path, source) where source is "vendored" or "system".

    Raises VendorError naming both routes when neither yields anything. The message is the whole
    point of the function: "llama-quantize not found" sends someone to a search engine, while
    naming the platform, the expected directory and the build step tells them what to do.
    """
    _log = log or (lambda _m: None)
    k = key or platform_key()

    if k:
        cand = VENDOR_BIN / k / (f"{name}.exe" if k.startswith("windows") else name)
        if _executable(cand):
            return cand, "vendored"

    if search_path:
        found = shutil.which(name)
        if found:
            _log(f"  note: using {name} from PATH ({found}) because this install carries no "
                 f"vendored copy for {k or 'this platform'}. Its version is whatever this machine "
                 f"has, which is not the pinned one, so record it beside any number it produces.")
            return Path(found), "system"

    where = f"{VENDOR_BIN / k}" if k else str(VENDOR_BIN)
    raise VendorError(
        f"{name} is not available. This install has no vendored copy at {where} and there is "
        f"none on PATH.\n"
        f"  * On a git checkout this is expected: the binaries are not committed. Run "
        f"`python tools/vendor_llama.py` to fetch the pinned build.\n"
        f"  * On an installed wheel it means no binary was built for "
        f"{k or platform.machine() + '/' + sys.platform}. Install llama.cpp so {name} is on PATH.\n"
        f"  * {name} is needed because k-quant quantisation exists only in llama.cpp's C++; the "
        f"gguf package can read a Q4_K but cannot produce one.")


def find_script(name):
    """A vendored pure-Python script from the conversion package.

    NOT committed, despite an earlier version of this message saying so: the package is 87 files
    and is fetched at build time, so a source checkout does not carry it until the vendoring tool
    has run. That distinction is the whole content of the failure below, because "you have not run
    a build step" and "your install is broken" want completely different things from a reader.
    """
    p = VENDOR_SRC / name
    if p.is_file():
        return p
    raise VendorError(
        f"the vendored script {name} is missing from {VENDOR_SRC}. It is fetched at build time "
        f"rather than committed, so in a source checkout run `python tools/vendor_llama.py` to "
        f"fetch it; in an installed wheel its absence is a packaging fault and should be reported.")


def make_executable(p):
    """Restore the executable bit, which neither a tar extraction nor a wheel install preserves.

    A wheel is a zip and zip does not carry POSIX modes in a way pip reinstates for data files, so
    a binary that shipped correctly arrives unrunnable. That failure looks like a missing file to
    everything except `ls -l`.
    """
    p = Path(p)
    mode = p.stat().st_mode
    p.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return p


def status(log=print):
    """What this install actually has, for a diagnostic line and for the pre-flight."""
    key = platform_key()
    manifest = load_manifest()
    out = {"platform": key, "pins": {}}
    for name, pin in manifest["pins"].items():
        out["pins"][name] = {"tag": pin["tag"], "published": pin["published"]}
    out["llama_quantize"] = None
    try:
        p, src = find_binary("llama-quantize", log=lambda _m: None)
        out["llama_quantize"] = {"path": str(p), "source": src}
    except VendorError:
        pass
    if log:
        log(f"platform: {key or 'unrecognised'}")
        for name, info in out["pins"].items():
            log(f"  pin {name}: {info['tag']} ({info['published'][:10]})")
        lq = out["llama_quantize"]
        log(f"  llama-quantize: {lq['source']} at {lq['path']}" if lq else
            "  llama-quantize: NOT AVAILABLE (quantisation will refuse)")
    return out
