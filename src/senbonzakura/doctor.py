# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
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

import argparse
import subprocess
import sys
from pathlib import Path

#: Exit codes. Distinguished so a CI job can treat a warning differently from a failure.
OK, WARN, FAIL = 0, 1, 2


class Check:
    """One question, its answer, and what to do about it."""

    def __init__(self, name, status, detail, fix=""):
        self.name, self.status, self.detail, self.fix = name, status, detail, fix

    @property
    def mark(self):
        return {"pass": "✓", "warn": "!", "fail": "✗"}[self.status]


def _pass(name, detail):
    return Check(name, "pass", detail)


def _warn(name, detail, fix=""):
    return Check(name, "warn", detail, fix)


def _fail(name, detail, fix=""):
    return Check(name, "fail", detail, fix)


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
        return _fail("llama-quantize", "not available", str(e).split(".")[0])
    # Runs, not exists. A binary missing a shared library exists and cannot start, which is exactly
    # what the first vendoring attempt produced.
    try:
        r = subprocess.run([str(exe), "--help"], capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        return _fail("llama-quantize", f"present at {exe} and will not run ({e})",
                     "re-run `python tools/vendor_llama.py`")
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
                     "re-run `python tools/vendor_llama.py`")
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
        names, broken = supported_architectures(script, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return [_fail("converter", f"will not run ({e})", "re-run `python tools/vendor_llama.py`")]

    out = []
    if not names:
        out.append(_fail("converter", "reported no supported architectures at all",
                         "the vendored package is incomplete; re-run the vendoring tool"))
        return out
    if broken:
        out.append(_fail(
            "architecture modules", f"{len(broken)} failed to import: {', '.join(broken[:6])}",
            "these architectures are ADVERTISED AND ABSENT. Usually a `gguf` package mismatch; "
            "re-run `python tools/vendor_llama.py` so gguf-py comes from the pinned tag"))
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

    `build_corpora.py` fetches through the GitHub CLI so that no credential is ever handled by
    the script. On a machine without `gh` the recommended command is one a person cannot run, and
    telling them to run it is worse than useless: they follow the instruction, it fails, and the
    instruction was ours. Checked here because this is where the recommendation is made.
    """
    import shutil
    base = "run `python tools/build_corpora.py` in a source checkout"
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
        return _pass("bundled track", f"{p.stat().st_size / 1024:.0f} KB")
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


def check_torch():
    try:
        import torch
    except ImportError:
        return _fail("torch", "not installed", "install the project's dependencies")
    if torch.cuda.is_available():
        try:
            free, _total = torch.cuda.mem_get_info()
            return _pass("torch", f"{torch.__version__}, cuda, "
                                  f"{torch.cuda.get_device_name(0)}, {free / 1e9:.1f} GB free")
        except Exception:
            return _pass("torch", f"{torch.__version__}, cuda")
    return _warn("torch", f"{torch.__version__}, no cuda device",
                 "editing a model on CPU works and is slow; scoring is fine")


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
    except ImportError:
        return _warn("pinned memory", "not measured, torch is missing",
                     "install torch; this number only matters for streaming a model larger "
                     "than the card")
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
    try:
        import torch
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    except ImportError as e:
        return [_warn("deep", f"skipped, {e}", "install torch and transformers to run it")]

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
        try:
            rc = convert.run([str(d / "m"), str(gguf), "--outtype", "bf16"], log=lambda _m: None)
        except SystemExit as e:
            return [*out, _fail("deep convert", f"failed: {e}", "see the message above")]
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


def run_checks(*, deep=False, log=print):
    checks = [check_platform(), check_torch()]
    # Cheap by default (one 256 MB probe: can this machine pin at all), thorough under --deep,
    # where finding the real ceiling means climbing to it.
    checks.append(check_pinned_memory(PINNED_DEEP_MAX_BYTES if deep else None))
    checks += check_pins()
    checks.append(check_quantize())
    checks += check_converter()
    checks.append(check_track())
    checks.append(check_table_io())
    checks += check_corpora()
    if deep:
        checks += deep_check(log=log)
    return checks


def report(checks, log=print):
    log("senbonzakura doctor")
    log("")
    width = max(len(c.name) for c in checks) + 2
    for c in checks:
        log(f"  {c.mark}  {c.name:<{width}} {c.detail}")
        if c.fix and c.status != "pass":
            log(f"     {' ' * width} -> {c.fix}")
    fails = [c for c in checks if c.status == "fail"]
    warns = [c for c in checks if c.status == "warn"]
    log("")
    log(f"  {len(checks)} checks, {len(checks) - len(fails) - len(warns)} pass, "
        f"{len(warns)} advisory, {len(fails)} failed")
    if fails:
        log("")
        log("  This install cannot do what it claims. Fix the failures above before a long run:")
        log("  finding this on a rented card, with the weights already loaded, costs money.")
        return FAIL
    return WARN if warns else OK


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="senbonzakura doctor",
        description="Check that this install can actually convert, quantise and measure.")
    ap.add_argument("--deep", action="store_true",
                    help="also build a two-layer model and take it through convert and quantise. "
                         "Slower, and the only check that proves the whole chain")
    a = ap.parse_args(argv)
    return report(run_checks(deep=a.deep))
