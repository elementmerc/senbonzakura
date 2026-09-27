# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Find a vendored third-party executable, or say exactly why there is not one.

WHAT IS VENDORED AND WHAT IS NOT, AND WHY THEY DIFFER

Two kinds of artefact are pinned in `vendor/pins.json`, and they are handled differently on
purpose:

  * `convert_hf_to_gguf.py` is **committed to the repository.** It is source, so vendoring it the
    way you vendor source means a reader can see in a diff exactly what changed when the pin moved.
    That is the whole value of vendoring a script rather than fetching one.
  * The **binaries are not committed.** Six platform archives is roughly eighty megabytes of
    opaque blob per pin bump, in a public repository, reviewable by nobody. They are downloaded at
    wheel-build time by `tools/packaging/vendor_llama.py` and travel inside the wheel instead.

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
import re
import shutil
import stat
import sys
from pathlib import Path

from .vendoring import VendorError, load_manifest

#: Where `tools/packaging/vendor_llama.py` places what it extracts, and where the wheel carries it.
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


#: What counts as runnable on Windows, which has no execute bit. `os.access(p, os.X_OK)` returns
#: True there for ANY readable file, so the POSIX test does not merely fail to help, it actively
#: says yes to a text file. Windows decides by extension instead.
_WINDOWS_EXECUTABLE_SUFFIXES = frozenset({".exe", ".bat", ".cmd", ".com"})


def _executable(p):
    """Whether this path is something the operating system will actually run.

    Two rules, because the two systems answer differently. On POSIX it is the execute bit. On
    Windows there is no such bit and `os.access(..., X_OK)` answers True for every readable file,
    so a downloaded README would have counted as a vendored binary; there it is the extension.
    """
    if not p.is_file():
        return False
    if sys.platform.startswith("win"):
        return p.suffix.lower() in _WINDOWS_EXECUTABLE_SUFFIXES
    return os.access(p, os.X_OK)


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
        f"`python tools/packaging/vendor_llama.py` to fetch the pinned build.\n"
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
        f"rather than committed, so in a source checkout run `python tools/packaging/vendor_llama.py` to "
        f"fetch it; in an installed wheel its absence is a packaging fault and should be reported.")


def make_executable(p):
    """Restore the executable bit, which neither a tar extraction nor a wheel install preserves.

    A wheel is a zip and zip does not carry POSIX modes in a way pip reinstates for data files, so
    a binary that shipped correctly arrives unrunnable. That failure looks like a missing file to
    everything except `ls -l`.
    """
    p = Path(p)
    if sys.platform.startswith("win"):
        # A documented no-op rather than a silent one. Windows has no execute bit to restore, and
        # chmod'ing S_IXUSR there changes nothing while reading as though it did.
        return p
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


#: Lines a quiet run passes through anyway, because they are the reason somebody would have wanted
#: the output. Matched case-insensitively against each line.
#:
#: THIS IS WHAT "STDERR ALWAYS THROUGH" MEANT. The operator's decision on the chatter (2026-09-27)
#: was to summarise by default with stderr passing through untouched, and that decision rested on a
#: premise that turned out to be false: the vendored converter calls `logging.basicConfig` with no
#: stream, so ITS 350 LINES ARE STDERR. Letting stderr through would have left the defect exactly
#: where it was, and llama-quantize's fallback warning is on stderr too, so the stream cannot be
#: passed through untouched AND scanned. Separating by stream was never going to work here; the
#: distinction that matters is what a line SAYS.
LOUD_LINES = (re.compile(r"\b(error|warning|failed|failure|cannot|unsupported|refus)", re.IGNORECASE),)

#: How often a suppressed run says it is still alive. A conversion of a large model runs for many
#: minutes; silence for that long is indistinguishable from a hang, and the reason people reach for
#: Ctrl+C on a job that was working.
HEARTBEAT_S = 30

#: How many of the tool's last lines are kept for a failure report. A refusal that says "the
#: converter exited 1" and nothing else makes suppression a downgrade, which is the trap in hiding
#: output at all: the chatter is noise right up until it is the only evidence.
KEEP_LINES = 25


def relay(stream, *, verbose, log, watch=(), always=LOUD_LINES,
          heartbeat_s=HEARTBEAT_S, keep=KEEP_LINES, now=None):
    """Pass a vendored tool's output through, or summarise it, and never lose what matters.

    WHY THIS EXISTS, 2026-09-27

    `convert` handed the vendored converter the terminal, so about 350 lines of per-tensor chatter
    arrived ahead of this tool's own five-line summary. A surface audit found the summary buried:
    every word of the output was true and the reader still had to scroll for the part written for
    them.

    Suppressing output is not free, and this function is mostly about the two ways it goes wrong.
    A long job that prints nothing looks hung, so a heartbeat names the tool's most recent line
    every `heartbeat_s`. A failure whose diagnostic was swallowed is worse than noise, so the last
    `keep` lines are always retained and handed back for the refusal to quote.

    `watch` is a sequence of compiled patterns whose matches are collected and returned. `quantise`
    reads the fallback warning that way: buried once, and it turned out to be the one line that
    contradicted this tool's own verdict.

    Returns (matches, tail): matches is a list of (pattern, match) in the order they were seen, tail
    is the last lines as a list.
    """
    import collections
    import time as _time

    clock = now or _time.monotonic
    tail = collections.deque(maxlen=keep)
    matches = []
    started = clock()
    last_beat = started
    lines = 0

    for line in stream:
        lines += 1
        tail.append(line.rstrip("\n"))
        for pattern in watch:
            found = pattern.search(line)
            if found:
                matches.append((pattern, found))
        if verbose:
            sys.stdout.write(line)
            continue
        # A WARNING OR AN ERROR IS NEVER SUMMARISED AWAY. See LOUD_LINES for why this is decided by
        # what the line says rather than by which stream it arrived on. To stderr, because that is
        # where it came from and a pipeline that separates the streams should keep getting it there.
        if any(pattern.search(line) for pattern in always):
            sys.stderr.write(line)
            continue
        # The clock is read ONCE per line, into a variable. Three separate calls per heartbeat is
        # three syscalls on every line of a run with thousands of them, and it made the interval
        # untestable without stubbing a clock that answers the same question three times.
        at = clock()
        if at - last_beat >= heartbeat_s:
            last_beat = at
            from . import say
            log(f"  still working, {lines} lines in, {at - started:.0f}s: "
                f"{say.shorten(line.strip(), 60)}")
    if verbose:
        sys.stdout.flush()
    return matches, list(tail)
