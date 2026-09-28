# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura doctor`: prove this install can actually do the things it claims.

WHY THIS EXISTS, AND IT IS ONE SPECIFIC BUG

On 2026-08-18 the vendored converter's `import gguf` was resolving to the PyPI package, which is
NOT the code inside llama.cpp's tree even though both declare version 0.19.0. The PyPI one lacks
constants the pinned converter uses, so `conversion/lfm2.py` raised on import, and
`--print-supported-models` went on advertising `Lfm2MoeForCausalLM` regardless, because that list
is built from a static registry rather than from what imported.

Two target models were unconvertible. Every surface reported success. It was found by hand, on the
day it mattered, by someone who happened to read a stderr line.

The pre-flight in `convert` now refuses a run when any architecture module failed to import, which
catches that instance at the moment of use. This catches the CLASS, and catches it before a rented
card is holding weights: it imports every module, runs the binary, loads the bundled data, and with
`--deep` converts and quantises a real two-layer model end to end.

WHAT A CHECK HERE HAS TO BE

Every check answers "does this WORK", not "is this PRESENT". A file that exists and cannot be
imported, a binary that exists and cannot start, and a corpus that is packed and does not decode
are the three shapes of failure this project has actually shipped, and all three pass a presence
test.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import argresolve, say

#: Exit codes. Distinguished so a CI job can treat a warning differently from a failure.
OK, WARN, FAIL = 0, 1, 2

#: The command's own title, in ONE place. It was a literal in `report` and another in `main`, kept
#: in step by hand through a `header=False` argument, which is two sources of truth about one
#: string. The parser's `prog` is the same name and reads it too.
TITLE = "senbonzakura doctor"

#: What the checks are a list of. A terminal run prints the bundled corpora's licence notice while
#: the checks are being collected, so without this the report began mid-paragraph in somebody
#: else's prose and a reader scanning for "is my install fine" had to find where one ended.
SECTION = "Installed:"


#: The tick and cross, unless the console cannot write them.
#:
#: THE PRE-FLIGHT COMMAND CRASHED ON WINDOWS. A Windows console is cp1252 by default, which has no
#: U+2713, so `doctor` died with a UnicodeEncodeError traceback partway through printing its own
#: report. The one command whose entire job is to tell somebody whether their install works was
#: the one that could not finish saying so, and it took the whole CI job down with it.
#:
#: Decided once, at import, from what stdout can actually encode. ASCII marks are still aligned and
#: still distinguishable; a traceback is neither.
#:
#: The ASCII pass mark is lower case, and that is not a style choice. `say.is_marker` treats a line
#: whose first word is `OK` as a machine marker, one of the four bare markers other tools grep for,
#: so an upper-case fallback made every passing check on a Windows console indistinguishable from a
#: protocol line. `ok` is the same width, reads the same, and is not a marker.
def _marks():
    glyphs = {"pass": "\u2713", "warn": "!", "fail": "\u2717"}
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "".join(glyphs.values()).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return {"pass": "ok", "warn": " !", "fail": "XX"}
    return glyphs


_MARKS = _marks()


class Check:
    """One question, its answer, and what to do about it."""

    def __init__(self, name, status, detail, fix="", group=None):
        self.name, self.status, self.detail, self.fix = name, status, detail, fix
        # Which run of related checks this belongs to, so the report can put a blank line where one
        # run ends. None means ungrouped, which is what a hand-built list in a test is, and an
        # ungrouped report is laid out exactly as it was before.
        self.group = group

    @property
    def mark(self):
        return _MARKS[self.status]


def _pass(name, detail):
    return Check(name, "pass", detail)


def _warn(name, detail, fix=""):
    return Check(name, "warn", detail, fix)


def _vendor_remedy():
    """The right advice for this install, which is not the same advice everywhere.

    `tools/packaging/vendor_llama.py` exists in a checkout and in no wheel, so telling an installed user to
    run it sends them looking for a file they do not have. Same shape as the `--track default`
    fault that `bundled.running_from_a_checkout()` was written for.
    """
    from . import bundled
    if bundled.running_from_a_checkout():
        return "re-run `python tools/packaging/vendor_llama.py`"
    return (f"this install's vendored converter is incomplete, which is a packaging fault rather "
            f"than something you can fix locally. Please report it at {bundled.ISSUES}")


def _fail(name, detail, fix=""):
    return Check(name, "fail", detail, fix)


#: How much of a failed sub-command's own output one check line carries. The report prints a check
#: on one line; the whole transcript belongs to running that command directly.
_LAST_WORDS_CHARS = 180


def _exit_detail(exc):
    """What a `SystemExit` actually said, as one line.

    `str(SystemExit(2))` is "2", which reads as a sentence and is a status. Telling them apart here
    keeps a check from offering a number as the reason something failed, which is the one fact the
    reader cannot act on.
    """
    code = exc.code
    if code is None or isinstance(code, int):
        return f"exit status {0 if code is None else code}"
    return " ".join(str(code).split())


def _last_words(lines, otherwise):
    """The last thing a sub-command said before it stopped, or `otherwise` when it said nothing.

    Whitespace is collapsed because this lands in a one-line report, and a captured line carrying
    its own newline would break the column the reader is scanning.
    """
    said = [" ".join(str(line).split()) for line in lines]
    said = [line for line in said if line]
    if not said:
        return otherwise
    return f"the last line it printed: {say.shorten(said[-1], _LAST_WORDS_CHARS)}"


def check_platform():
    from .vendored import platform_key
    key = platform_key()
    if not key:
        return _warn("platform", f"unrecognised ({sys.platform})",
                     "vendored binaries cannot be selected; PATH copies will be used if present")
    return _pass("platform", key)


def check_pins():
    from .vendoring import load_manifest
    try:
        m = load_manifest()
    except Exception as e:
        return [_fail("pins", f"cannot be read ({e})", "reinstall, or check the wheel's data files")]
    out = []
    for name, pin in (m.get("pins") or {}).items():
        out.append(_pass(f"pin {name}", f"{pin['tag']} ({pin['published'][:10]})"))
    return out or [_fail("pins", "the manifest names no pins", "the manifest is empty or malformed")]


def check_quantize():
    from .vendored import VendorError, find_binary
    try:
        exe, source = find_binary("llama-quantize", log=lambda _m: None)
    except VendorError as e:
        # AN ADVISORY, NOT A FAILURE, and the remedy says what it is for.
        #
        # Six of seven readers in the 2026-09-26 user pass reported this same check independently,
        # and they reported two things about it. First, the remedy was `str(e).split(".")[0]`, which
        # is the error's own first clause, so the one FAILING line in the report restated itself
        # while every advisory beside it taught something. Second, and worse: a missing quantiser
        # made `doctor` print "This install cannot do what it claims" and exit 2, at people who had
        # installed the tool correctly and only wanted to measure.
        #
        # It is not a failure, because the universal wheel carries no binaries BY DESIGN: PyPI
        # refuses a `linux_x86_64` tag, so the portable wheel ships without them and `RELEASING.md`
        # publishes both. An install behaving exactly as documented must not be reported as broken.
        # `convert` and `quantise` are the only two commands that need it; everything else, the
        # whole measurement side included, works without it.
        #
        # The remedy names where the binaries actually come from. An earlier draft of this fix said
        # `senbonzakura fetch llama-cpp`, suggested by one of the readers and repeated by me without
        # checking: `fetch` downloads a MODEL FILE and has nothing to do with llama.cpp. A wrong
        # remedy is worse than the tautology it replaced, because it sends somebody somewhere.
        del e                     # its first clause is the check's own detail line, printed above
        return _warn("llama-quantize", "not available",
                     "only `senbonzakura convert` and `senbonzakura quantise` need it; measuring "
                     "and abliterating do not. On Linux with glibc 2.35 or newer the manylinux "
                     "wheel ships the binaries, so `pip install --force-reinstall senbonzakura` "
                     "picks them up; otherwise build llama.cpp yourself and put `llama-quantize` "
                     "on PATH")
    # Runs, not exists. A binary missing a shared library exists and cannot start, which is exactly
    # what the first vendoring attempt produced.
    try:
        r = subprocess.run([str(exe), "--help"], capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        return _fail("llama-quantize", f"present at {exe} and will not run ({e})",
                     "re-run `python tools/packaging/vendor_llama.py`")
    out = (r.stdout + r.stderr).decode("utf-8", errors="replace").lower()
    # A MISSING SHARED LIBRARY IS NOT A WRONG BINARY, and saying so sends the reader to re-vendor
    # a file that is already correct. llama.cpp links against OpenMP, and a slim image or a
    # minimal host does not carry it: the binary is exactly what we think and the system is not.
    # Seen for real inside `python:3.13-slim`, twice, and misdiagnosed both times.
    if "error while loading shared libraries" in out or "cannot open shared object" in out:
        missing = out.split("error while loading shared libraries:")[-1].split(":")[0].strip()
        return _fail("llama-quantize",
                     f"present at {exe} and cannot start: a shared library is missing"
                     + (f" ({missing})" if missing else ""),
                     "install the library it names. On Debian and Ubuntu the usual one is "
                     "libgomp1: `apt-get install -y libgomp1`. Re-vendoring will not help; the "
                     "binary is correct and the system is missing a dependency of it")
    if "usage" not in out:
        return _fail("llama-quantize", "ran and printed no usage; the binary is not what we think",
                     "re-run `python tools/packaging/vendor_llama.py`")
    return _pass("llama-quantize", f"{source}, runs")


def check_converter(timeout=300):
    """The converter, and EVERY architecture module it claims to provide.

    The second half is the point. A module that raises on import leaves its architectures on the
    advertised list, so "supported" and "works" are different questions and only one of them was
    ever being asked.
    """
    from .convert import supported_architectures
    from .vendored import VendorError, find_script
    try:
        script = find_script("convert_hf_to_gguf.py")
    except VendorError as e:
        return [_fail("converter", "not vendored", str(e).split(";")[-1].strip())]

    try:
        names, broken, died = supported_architectures(script, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return [_fail("converter", f"will not run ({e})", _vendor_remedy())]

    out = []
    if died:
        # THE SCRIPT DIED, which is a different fault from the script running and listing nothing.
        # Reported as incomplete vendoring until 2026-09-10, on an install whose only problem was
        # a missing torch, with a remedy naming a file no wheel contains.
        out.append(_fail("converter", f"could not start: {died}",
                         "that is the converter failing to import, not a vendoring problem. If it "
                         "names a missing module, install it: the converter imports torch, numpy "
                         "and gguf at module level."))
        return out
    if not names:
        out.append(_fail("converter", "ran and reported no supported architectures at all",
                         _vendor_remedy()))
        return out
    if broken:
        out.append(_fail(
            "architecture modules", f"{len(broken)} failed to import: {say.some_of(broken, 6)}",
            "these architectures are ADVERTISED AND ABSENT. Usually a `gguf` package mismatch; "
            "re-run `python tools/packaging/vendor_llama.py` so gguf-py comes from the pinned tag"))
    else:
        out.append(_pass("architecture modules", f"all import, {len(names)} architectures"))

    # Named because they are what this tool is pointed at, and because the LFM2 pair is what the
    # gguf mismatch took out. A future target added here fails loudly rather than at use.
    #
    # THE `broken` GUARD IS NOT DECORATION. `names` comes from the converter's static registry, so
    # while any module is failing to import, an architecture appearing in it means the registry
    # mentions it, NOT that it works. Without this branch, doctor reported the architecture
    # modules as failing and an LFM2 architecture as supported three lines apart, which is the
    # very defect this command exists to catch, reproduced by the command itself. Found by
    # removing the vendored gguf-py and reading doctor's own output.
    for arch in ("Lfm2ForCausalLM", "Lfm2MoeForCausalLM", "Qwen3ForCausalLM", "LlamaForCausalLM"):
        if broken:
            out.append(_warn(f"arch {arch}", "cannot be confirmed while modules fail to import",
                             "the registry lists it; that is not the same as it working"))
        elif arch in names:
            out.append(_pass(f"arch {arch}", "supported"))
        else:
            out.append(_warn(f"arch {arch}", "not supported at this pin",
                             "refresh the pin at the next release"))
    return out


def _corpus_fix():
    """The instruction to hand someone whose corpora are missing, including what it needs.

    `senbonzakura corpora` fetches through the GitHub CLI so that no credential is ever handled by
    the tool. On a machine without `gh` the recommended command is one a person cannot run, and
    telling them to run it is worse than useless: they follow the instruction, it fails, and the
    instruction was ours. Checked here because this is where the recommendation is made.
    """
    import shutil
    base = "run `senbonzakura corpora`"
    if shutil.which("gh") is None:
        return base + " (it needs the GitHub CLI, which is not installed: https://cli.github.com)"
    return base


def check_corpora():
    from .corpora import CORPORA, CorpusError, load
    out = []
    for key in sorted(CORPORA):
        try:
            rows = load(key)
        except CorpusError as e:
            out.append(_fail(f"corpus {key}", str(e).split(".")[0], _corpus_fix()))
            continue
        expected = CORPORA[key].rows
        out.append(_pass(f"corpus {key}", f"{len(rows)} prompts")
                   if len(rows) == expected else
                   _fail(f"corpus {key}", f"{len(rows)} prompts, expected {expected}",
                         "the pack and the code disagree; rebuild it"))
    return out


def check_track():
    try:
        from . import bundled
        p = Path(bundled.data_path())
        if not p.is_file():
            return _fail("bundled track", "not installed", "reinstall, or rebuild it")
        # THE PATH, not just the size. Auditing the bundled track is a real task and it is the
        # one that decides a rented-card booking, and the only way to find where it unpacks to
        # was to import `senbonzakura.bundled` and call `ensure()` in a REPL. Reading the source
        # is precisely what a discoverable tool must not require, and `doctor` is where somebody
        # already is when they ask this.
        return _pass("bundled track",
                     f"{p.stat().st_size / 1024:.0f} KB, unpacks to {bundled.cache_dir()}")
    except Exception as e:
        return _fail("bundled track", f"cannot be read ({e})", "reinstall")


def check_table_io():
    """Can this install read and write a track at all?

    THE CHECK THAT WAS MISSING THE MOMENT PYARROW BECAME THE ONLY BASE DEPENDENCY. `doctor`
    exists to say what an install cannot do, and with pyarrow and datasets both removed it
    reported 16 passes, 2 advisories, 0 failures and exited 0, while `track --audit` and
    `track` build both died in the same environment. Nothing in the roster touched the table
    reader: the bundled-track check stats a packed blob and the corpora check unpacks one, and
    neither goes near Arrow.

    Written as a round trip rather than an import, because "the package is present" is the kind
    of check this project has already been caught by twice.
    """
    import tempfile

    from . import trackio
    try:
        with tempfile.TemporaryDirectory() as d:
            table = Path(d) / "probe"
            trackio.write_text_column(table, ["a prompt", "another"])
            got = trackio.read_text_column(table)
        if got != ["a prompt", "another"]:
            return _fail("track tables", f"a round trip returned {got!r}",
                         "reinstall; the table reader and writer disagree")
        return _pass("track tables", "written and read back")
    except Exception as e:
        return _fail("track tables", f"cannot be read or written ({e})",
                     "pip install --force-reinstall senbonzakura")


#: Windows refusing to load a DLL because an Application Control policy says so. Smart App
#: Control is ON by default on a new Windows 11 installation and blocks unsigned binaries, and
#: torch ships a lot of unsigned DLLs.
_WINDOWS_APP_CONTROL = 4551


def check_torch():
    try:
        import torch
    except ImportError:
        return _fail("torch", "not installed", "install the project's dependencies")
    except OSError as e:
        # FOUND ON REAL WINDOWS HARDWARE, 2026-09-17. `except ImportError` alone was not enough:
        # a DLL that is present and refuses to LOAD raises OSError, which is not an ImportError,
        # so it went straight past this handler and out through `main` as a thirty line traceback.
        #
        # That is the worst place in the tool for it to happen. `doctor` exists to say whether
        # this install can do the job, the user runs it precisely because something is wrong, and
        # on a stock Windows 11 box it died instead of answering. `pip install` had reported
        # success on both distributions and `senbonzakura --version` worked, because the entry
        # point deliberately parses without importing torch, so every cheap signal said the
        # install was fine.
        if getattr(e, "winerror", None) == _WINDOWS_APP_CONTROL:
            return _fail(
                "torch",
                "installed, and Windows will not let it load: an Application Control policy "
                f"blocked one of its DLLs ({e})",
                "This is Smart App Control, which is ON by default on a new Windows 11 "
                "installation and blocks unsigned binaries. torch ships many. Two ways out, and "
                "they are not equal: run this under WSL2 instead, which is unaffected and is how "
                "this project is developed; or turn Smart App Control off in Settings, Privacy "
                "and security, Windows Security, App and browser control. Read the second one "
                "twice: Smart App Control cannot be turned back ON without reinstalling Windows.")
        return _fail("torch", f"installed, and it will not load ({e})",
                     "the package is present and its native libraries cannot be loaded. Reinstall "
                     "it with `pip install --force-reinstall torch`, and if that does not help, "
                     "the fault is in the environment rather than in the package.")
    if torch.cuda.is_available():
        try:
            free, _total = torch.cuda.mem_get_info()
            return _pass("torch", f"{torch.__version__}, cuda, "
                                  f"{torch.cuda.get_device_name(0)}, {free / 1e9:.1f} GB free")
        except Exception:
            return _pass("torch", f"{torch.__version__}, cuda")
    # WHAT THIS LINE USED TO SAY AND WHY IT WAS WRONG, found by a reader 2026-09-27. It read
    # "editing a model on CPU works and is slow", which describes the wrong outcome: on the default
    # flags an edit on CPU does not start at all. `capability.refuse_slow_cpu_probe` prices the
    # default 200-item, 512-token probe at roughly four hours of host generation and exits rather
    # than spending it. An advisory that promises a slow success where the tool delivers a refusal
    # sends the reader looking for a fault in their install.
    return _warn("torch", f"{torch.__version__}, no cuda device",
                 "scoring is fine on CPU. An edit is not, on the default flags: the capability "
                 "probe costs about four hours at host speed, so the run refuses rather than "
                 "starting, and names the ways past it. They are --capability-n 0 for no "
                 "capability number, or --slow-probe-ok to spend the hours. The edit itself "
                 "works on CPU, slowly.")


#: Steps for the page-locked probe. Big enough that the per-allocation overhead does not dominate,
#: small enough that the answer is not rounded to uselessness on a machine with a low ceiling.
PINNED_STEP_BYTES = 256 * 1024 * 1024

#: How far the shallow probe goes. One step: enough to answer "can this machine pin at all", which
#: is the question a default `doctor` run should cost nothing to answer.
PINNED_SHALLOW_STEPS = 1

#: How far `--deep` will climb. A ceiling on the ceiling-finder, because the honest way to find the
#: limit is to reach it, and reaching it on a machine with a huge one would mean pinning most of
#: host RAM. Better to report "at least 8 GB" than to make a laptop unusable establishing 30.
PINNED_DEEP_MAX_BYTES = 8 * 1024 * 1024 * 1024


def _measure_pinned_ceiling(max_bytes):
    """How much page-locked host memory this machine will actually hand out, measured.

    WHY A DOCTOR CHECK CARES, which is not obvious.

    Streaming a model through the card layer by layer only pays off if the host-to-device copy
    OVERLAPS with compute, and `copy_(non_blocking=True)` only overlaps when the source is
    page-locked. Above the machine's page-locked ceiling the allocation quietly falls back to
    ordinary pageable memory, the copy becomes synchronous, and the overlap is lost. The observed
    cost of crossing that line elsewhere was GPU utilisation dropping from 100% to 79.3%, with
    nothing in any log to say why.

    So this is a number a streaming run has to be designed against, and the failure it prevents is
    the kind that gets blamed on streaming being slow rather than on a host store being too big.

    WHY IT ALLOCATES RATHER THAN READING A LIMIT. `ulimit -l` (RLIMIT_MEMLOCK) looks like the
    answer and is not. Measured on the reference machine 2026-09-02: the rlimit reads 64 MB and the
    probe pinned 8.6 GB without complaint, because CUDA's host allocator does not go through the
    path that limit governs. A check that read the rlimit would have reported a ceiling 134 times
    too low and sent a streaming design chasing a constraint that is not there.

    Returns (bytes_reached, hit_limit, note). `hit_limit` is False when the probe stopped because
    it ran out of budget rather than because the machine refused, so "at least this much" and
    "exactly this much" are never confused.
    """
    import torch
    blocks, reached = [], 0
    try:
        while reached + PINNED_STEP_BYTES <= max_bytes:
            try:
                blocks.append(torch.empty(PINNED_STEP_BYTES, dtype=torch.uint8, pin_memory=True))
            except (RuntimeError, MemoryError) as e:
                return reached, True, type(e).__name__
            reached += PINNED_STEP_BYTES
        return reached, False, ""
    finally:
        # Freed before returning, whatever happened. Holding gigabytes of page-locked memory past
        # the end of a diagnostic would be a worse bug than the one being diagnosed.
        blocks.clear()
        import gc
        gc.collect()


def check_pinned_memory(max_bytes=None):
    """Report the page-locked ceiling, or say plainly why it could not be measured."""
    try:
        import torch
    # OSError as well as ImportError: a torch that is present and will not LOAD raises OSError,
    # and on Windows with Smart App Control that is the common case rather than an exotic one.
    # `check_torch` above reports the cause properly; every check after it only has to survive.
    except (ImportError, OSError) as e:
        return _warn("pinned memory", f"not measured, torch is unusable ({type(e).__name__})",
                     "see the torch line above; this number only matters for streaming a model "
                     "larger than the card")
    if not torch.cuda.is_available():
        return _warn("pinned memory", "not measured, no cuda device",
                     "page-locked memory is only useful for overlapping host-to-device copies, "
                     "so there is nothing to measure without a card")
    budget = PINNED_STEP_BYTES * PINNED_SHALLOW_STEPS if max_bytes is None else max_bytes
    try:
        reached, hit_limit, note = _measure_pinned_ceiling(budget)
    except Exception as e:                     # a probe must never be the thing that fails a run
        return _warn("pinned memory", f"not measured ({type(e).__name__}: {e})",
                     "this is diagnostic only and does not affect an ordinary run")
    gb = reached / 1e9
    if reached == 0:
        return _warn("pinned memory", f"cannot pin even {PINNED_STEP_BYTES / 1e6:.0f} MB ({note})",
                     "a streaming run would lose host-to-device overlap entirely here; check "
                     "`ulimit -l` and how much host RAM is free")
    if hit_limit:
        return _pass("pinned memory", f"ceiling {gb:.1f} GB (refused more: {note}). A host-side "
                                      f"store above this loses copy/compute overlap")
    return _pass("pinned memory", f"at least {gb:.1f} GB, not probed further")


def deep_check(log=print):
    """Build a two-layer model, convert it, quantise it. The only check that proves the chain.

    Small enough to be quick and real enough that nothing about it is a mock: a genuine
    transformers checkpoint through the genuine converter and the genuine quantiser, verified by
    reading the header of what came out.
    """
    import tempfile

    out = []
    # IMPORTED ONE AT A TIME so the skip names the package that failed. Together, under one
    # `except`, a broken transformers beside a perfectly good torch was reported as "install torch
    # and transformers ... see the torch line above", and the torch line above reads as a pass.
    # OSError beside ImportError for the same reason `check_pinned_memory` does it: a package that
    # is present and will not LOAD raises OSError, which is the common shape on Windows.
    try:
        import torch
    except (ImportError, OSError) as e:
        return [_warn("deep", f"skipped, torch is unusable ({type(e).__name__}: {e})",
                      "the torch line above says why it will not load; fix that and this check "
                      "runs")]
    try:
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    except (ImportError, OSError) as e:
        return [_warn("deep", f"skipped, transformers is unusable ({type(e).__name__}: {e})",
                      "torch itself loaded here, so the torch line above does not explain this. "
                      "Repair transformers with: pip install --force-reinstall senbonzakura")]

    with tempfile.TemporaryDirectory(prefix="senbon-doctor-") as td:
        d = Path(td)
        try:
            tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        except Exception as e:
            return [_warn("deep", f"skipped, no tokenizer available offline ({type(e).__name__})",
                          "the deep check needs one small tokenizer; run it once with a network")]
        cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=16, max_position_embeddings=128)
        Qwen3ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(d / "m")
        tok.save_pretrained(d / "m")

        from . import convert
        gguf = d / "m.gguf"
        # KEPT RATHER THAN DISCARDED. This was `log=lambda _m: None` with the failure line saying
        # "see the message above": the converter's own words went nowhere, and a SystemExit raised
        # before the subprocess starts prints nothing anywhere at all, so the one line naming the
        # cause was thrown away by the code that then pointed at it.
        said = []
        try:
            rc = convert.run([str(d / "m"), str(gguf), "--outtype", "bf16"], log=said.append)
        except SystemExit as e:
            return [*out, _fail("deep convert", f"failed: {_exit_detail(e)}",
                                _last_words(said, "the converter printed nothing before it "
                                                  "stopped"))]
        if rc != 0 or not gguf.is_file():
            return [*out, _fail("deep convert", "produced no file", "")]

        from . import gguf_io
        head = gguf_io.verify(gguf, expect_quant="BF16")
        out.append(_pass("deep convert", f"{head['tensor_count']} tensors, {head['architecture']}"))

        from . import quantise
        q = d / "m-Q4_K_M.gguf"
        try:
            rc = quantise.run([str(gguf), str(q), "--type", "Q4_K_M"], log=lambda _m: None)
        except SystemExit as e:
            return [*out, _fail("deep quantise", f"failed: {e}", "")]
        if rc != 0 or not q.is_file():
            return [*out, _fail("deep quantise", "produced no file", "")]
        qh = gguf_io.verify(q, expect_quant="Q4_K_M")
        out.append(_pass("deep quantise", f"Q4_K_M, {qh['tensor_count']} tensors"))
    return out


def _in_group(checks, name):
    """Stamp a run of checks with the group it belongs to, and hand it back unchanged otherwise.

    The groups are the ones this report has always had; naming them only lets the renderer put a
    blank line where one ends. No check moves, and no check's verdict, wording or order changes.
    """
    for c in checks:
        c.group = name
    return checks


def run_checks(*, deep=False, log=print):
    # Cheap by default (one 256 MB probe: can this machine pin at all), thorough under --deep,
    # where finding the real ceiling means climbing to it.
    checks = _in_group([check_platform(), check_torch(),
                        check_pinned_memory(PINNED_DEEP_MAX_BYTES if deep else None)], "machine")
    checks += _in_group([*check_pins(), check_quantize(), *check_converter()], "tools")
    checks += _in_group([check_track(), check_table_io(), *check_corpora()], "data")
    if deep:
        checks += _in_group(deep_check(log=log), "deep")
    return checks


#: The narrowest a value column may be before the row stacks instead. Below this a value is
#: broken into two and three word fragments, which is harder to read than a second line.
_NARROWEST_VALUE = 24


def _row(log, prefix, body):
    """One `label  value` row, folded under its own label rather than off the edge.

    The continuation is indented to the width of the prefix, so a wrapped value stays a block
    under its own heading instead of colliding with the next row's label.
    """
    from . import say

    columns = say.width()
    room = columns - len(prefix)
    if room >= _NARROWEST_VALUE:
        lines = say.lines(str(body), columns=room) or [""]
        log(f"{prefix}{lines[0]}".rstrip())
        for line in lines[1:]:
            log(f"{' ' * len(prefix)}{line}")
        return
    # STACKED, because two columns do not fit. The label column is as wide as the longest check
    # name, so on a narrow terminal there is no room left for a value beside it and holding the
    # layout means running off the edge instead. The value drops to its own indented line, which
    # is the same shape `doctor` already uses for a fix.
    log(prefix.rstrip())
    for line in say.lines(str(body), columns=max(_NARROWEST_VALUE, columns - 5)) or [""]:
        log(f"     {line}")


def report(checks, log=print, *, advisories_ok=False, header=True):
    # `header=False` when the caller has already named the command, which on a terminal the banner
    # does. Default True so every other caller, and every test that renders a report on its own,
    # still gets the title.
    if header:
        log(TITLE)
    # The heading and the blank line around it run either way. The bundled corpora print their
    # licence notice while the checks are being collected, and that obligation stays, so this is
    # what marks where it ends and the answer to "is my install fine" begins.
    log("")
    log(SECTION)
    log("")
    width = max(len(c.name) for c in checks) + 2
    group = None
    for i, c in enumerate(checks):
        if i and c.group != group:
            log("")
        group = c.group
        # WRAPPED, AND THE INDENT PASSED RATHER THAN BAKED IN. `say` leaves an already indented
        # line alone deliberately, because an indented line is usually a command somebody has to
        # paste, so a fix written as `"     ...text"` went out unwrapped however long it was. One
        # of these runs to about three hundred characters, and a journey driving `doctor` in a
        # fifty column window is what found it.
        _row(log, f"  {c.mark}  {c.name:<{width}} ", c.detail)
        if c.fix and c.status != "pass":
            _row(log, f"     {' ' * width} -> ", c.fix)
    fails = [c for c in checks if c.status == "fail"]
    warns = [c for c in checks if c.status == "warn"]
    log("")
    log(f"  {len(checks)} checks, {len(checks) - len(fails) - len(warns)} pass, "
        f"{len(warns)} advisory, {len(fails)} failed")
    if fails:
        log("")
        _row(log, "  ", "This install cannot do what it claims. Fix the failures above before a "
                         "long run: finding this on a rented card, with the weights already "
                         "loaded, costs money.")
        return FAIL
    if warns:
        if advisories_ok:
            # ASKED FOR EXPLICITLY, so it is stated rather than silently different. A reader
            # comparing two logs has to be able to see why one exited 0 and the other 1.
            log("")
            _row(log, "  ", f"Exit status {OK}: advisories only, and --advisories-ok was given. "
                            f"Nothing failed.")
            return OK
        # SAY WHAT THE EXIT CODE MEANS, because the line above says "0 failed" and this returns 1.
        #
        # Reported from a GPU box on 2026-09-11 as a possible defect: doctor exited 1 while its own
        # summary said nothing had failed, on a machine whose only complaint was an absent CUDA
        # device. The convention is deliberate and documented at the top of this module, so an
        # advisory can be told from a failure by a CI job. But a reader has the summary and the
        # exit code and nothing connecting them, and every user on a GPU-less box meets this on
        # their first run. The fix is one sentence, not a changed convention: silencing the exit
        # code would remove the distinction the codes exist for.
        log("")
        _row(log, "  ", f"Exit status {WARN}: advisories only, nothing failed. This install "
                        f"works; the lines above are things it cannot do.")
        _row(log, "  ", f"({OK} means nothing to report, {WARN} advisories, {FAIL} something "
                        f"failed.)")
        _row(log, "  ", "Pass --advisories-ok to exit 0 here; a real failure still exits 2.")
    return WARN if warns else OK


def main(argv=None):
    ap = argresolve.ParserThatNamesUnknownFlags(
        allow_abbrev=False,
        prog=TITLE,
        description="Check that this install can actually convert, quantise and measure.")
    ap.add_argument("--deep", action="store_true",
                    help="also build a two-layer model and take it through convert and quantise. "
                         "Slower, and the only check that proves the whole chain")
    ap.add_argument("--advisories-ok", action="store_true",
                    help="exit 0 when the only complaints are advisories. For a CI step on a "
                         "machine that is not meant to have a GPU: without it a healthy CPU-only "
                         "install exits 1, because an advisory means this install cannot do "
                         "something. A genuine failure still exits 2 either way")
    a = ap.parse_args(argv)
    # NO TITLE OF ITS OWN. It used to print one here so the corpora's licence notice could not land
    # above it, which left the tool's name in the first six lines three times: twice from the
    # banner the entry point draws and once from here. The corpora block is now one scannable line
    # per corpus and the checks announce themselves with their own heading, so this line was
    # repetition rather than orientation.
    return report(run_checks(deep=a.deep), advisories_ok=a.advisories_ok, header=False)


if __name__ == "__main__":   # pragma: no cover
    from .entry import module_entry
    module_entry(main)
