#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""What a freshly installed senbonzakura must and must not be able to do.

Runs INSIDE a disposable container against a wheel installed from scratch, with no repository, no
network, no credentials and no vendored binaries. It answers a question the repository cannot ask
of itself: **a checkout has the binaries, the corpora sources and the git history sitting on disk,
so every check run there is run on a machine that already has what a stranger does not.**

The gap this was written after: the wheel ships no llama.cpp binaries, so a `pip install` cannot
convert or quantise. Nothing said so. It was found by hand, on a second machine, by running
`doctor` and reading the output.

THE POINT IS THE REFUSALS, NOT THE SUCCESSES.

A fresh install is *supposed* to be unable to quantise. What it must never do is pretend
otherwise, crash with a traceback, or exit 0 while saying it cannot work. So most of what follows
asserts the shape of a failure rather than the presence of a feature.

    python clean_room_checks.py [--expect-torch]

Exit 0 when the install behaves as a fresh install should, 1 otherwise, with every failure named.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys

FAILURES: list[str] = []
CHECKS = 0


def check(name, condition, detail=""):
    # A module-level tally, because this is a
    # single-pass script whose whole output is one running count.
    global CHECKS  # noqa: PLW0603
    CHECKS += 1
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}{(': ' + detail) if detail else ''}")
        FAILURES.append(name)
    return bool(condition)


def note(message):
    """Something the reader has to know that is not a pass or a fail.

    A code-only wheel legitimately carries no corpora, and calling that a failure would make
    the gate cry wolf on every CI run. Calling it a pass would let a release wheel ship empty.
    So it is neither, and it is loud.
    """
    print(f"  NOTE  {message}")


#: The phrases `doctor` uses when a binary is correct and the machine under it is not.
#:
#: Matched on more than one spelling deliberately. A guard that covers one spelling of a defect
#: reports confidently on the others, which is this project's most recurring failure shape, and
#: `doctor` has several ways of saying a binary could not start.
IMAGE_IS_AT_FAULT = (
    "shared library is missing",
    "error while loading shared libraries",
    "cannot open shared object",
)


def blames_the_image(verdict):
    """Whether this verdict is about the machine rather than about the wheel.

    Kept as a named function with its own test rather than inlined, because the whole point of
    row 8.7 is that the clean room was confidently wrong about whose fault something was, and a
    one-line condition inside a branch nobody can reach without a container is a condition
    nobody can watch being right.
    """
    lowered = str(verdict).lower()
    return any(phrase in lowered for phrase in IMAGE_IS_AT_FAULT)


def run(args, timeout=120):
    """Run the installed console script, never a checkout."""
    return subprocess.run([sys.executable, "-m", "senbonzakura", *args],
                          capture_output=True, text=True, timeout=timeout, check=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--expect-torch", action="store_true",
                    help="the full install: torch is present, so the paths needing it must work")
    a = ap.parse_args(argv)

    print("clean room: what this install can actually do\n")

    # ── BEFORE ANY CHECK: is this a clean room at all ────────────────────────────
    #
    # A clean room that is not clean is the one kind of test whose pass means nothing, and this
    # has happened here: the harness on one machine shadowed an installed 0.3.0 with a
    # `PYTHONPATH` pointing at a checkout, so what it tested was not what it installed. Every
    # check below then described the checkout, including the ones about what a wheel ships, and
    # the run reported on an artefact nobody built.
    #
    # IT REFUSES RATHER THAN UNSETTING, and that is the whole decision. Quietly dropping the
    # variable would make the run correct and leave the caller believing they had measured the
    # thing they pointed it at, so the next person wires it up the same way. Loudly refusing
    # costs one message and is the only version that changes anything.
    #
    # EXIT 2, not 1. One means the install misbehaved, which is a finding about the software;
    # this means the room was not sealed, which is a finding about the harness, and a caller
    # that treats them alike learns to ignore both. The gate this project ships makes exactly
    # that distinction and it would be strange to not make it here.
    unclean = []
    if os.environ.get("PYTHONPATH"):
        unclean.append(
            f"PYTHONPATH is set to {os.environ['PYTHONPATH']!r}. Anything on it shadows the "
            f"installed wheel, so these checks would describe whatever is on that path.")
    # PYTHONHOME IS DELIBERATELY NOT CHECKED, and that was tried first. A value that would do any
    # harm stops the interpreter before this file is reached ("No module named 'encodings'",
    # measured), and a value that does no harm is a set variable pointing at the prefix already in
    # use, which this would refuse for nothing. A guard that cannot fire for the dangerous case
    # and does fire for the harmless one is worse than no guard: it teaches the reader that the
    # refusal is noise.
    spec = importlib.util.find_spec("senbonzakura")
    origin = spec.origin if spec else ""
    if "site-packages" not in (origin or ""):
        unclean.append(
            f"senbonzakura imports from {origin or 'nowhere'}, which is not an installed "
            f"location. A checkout on the path answers every question below about itself.")
    if unclean:
        print("  REFUSED  this is not a clean room, so nothing was checked.\n")
        for reason in unclean:
            print(f"    - {reason}")
        print("\n  Nothing here was unset for you: a run that silently repaired its own "
              "environment would pass while measuring something else, and the caller would "
              "wire it up the same way next time. Clear the variable, or install the wheel into "
              "the environment you are pointing this at, and run it again.")
        print("\n  A REFUSAL IS NOT A PASS AND NOT A FAILURE OF THE INSTALL. Exit 2.")
        return 2

    check("installed from a wheel, not a checkout", "site-packages" in (origin or ""),
          f"imported from {origin}")
    check("no repository present", not __import__("pathlib").Path("/src/.git").exists())

    have_torch = importlib.util.find_spec("torch") is not None
    check("torch presence matches what was asked for", have_torch == a.expect_torch,
          f"torch {'present' if have_torch else 'absent'}")

    # ── a delegated command starts without the deep-learning stack ───────────────
    # Tested FIRST, and it is the check a --no-deps install can actually make.
    #
    # `doctor` exists to tell you an install is incomplete, so it has to be able to run on one.
    # It used to reach the tool through `cli`, which imports torch, optuna and transformers at
    # module level, so the one command whose job is to diagnose a missing torch could not start
    # without torch. The dispatcher in `entry.py` is what fixed that, and this is the only place
    # that can prove it, because every other machine we own already has torch.
    r = run(["doctor"])
    out = r.stdout + r.stderr
    check("doctor starts with no torch and no optuna installed", "checks," in out, out[-400:])

    if a.expect_torch:
        r2 = run(["--help"])
        check("--help exits 0", r2.returncode == 0, f"exit {r2.returncode}")
        check("--help names the commands", "quantise" in r2.stdout and "doctor" in r2.stdout)
    else:
        # Bare `--help` is the abliterate parser, and abliteration genuinely cannot proceed
        # without torch. Its absence here is correct; said out loud so the gap is not read as one.
        print("\n  note  bare --help is the abliterate parser, which needs torch. Not checked "
              "in this mode.\n")

    # ── doctor tells the truth about THIS wheel ──────────────────────────────────
    #
    # Two kinds of wheel exist and they have opposite correct answers, so the expectation is read
    # off the artefact rather than hard-coded. Hard-coding it would mean the checks quietly
    # asserted a stale truth the day platform wheels shipped, which is the failure mode this file
    # was written to catch in the software.
    check("doctor names the quantiser either way", "llama-quantize" in out)

    # THREE states, and they are read from the FILESYSTEM and doctor's status glyph rather than
    # from its prose.
    #
    # The first version of this matched the sentence "vendored, runs", which made a wheel shipping
    # broken binaries indistinguishable from one shipping none. Widening it to also match "will
    # not run" missed a THIRD phrasing ("ran and printed no usage"), because doctor has several
    # ways to say a binary is unusable and matching prose means chasing all of them forever. That
    # is precisely the failure `events.py` exists to avoid, committed here by the same hand that
    # wrote that docstring.
    #
    # So: does the file exist (a fact), and did doctor pass it (a glyph)?
    import pathlib as _pl
    spec2 = importlib.util.find_spec("senbonzakura")
    if not (spec2 and spec2.origin):
        # Guarded like the check twenty lines above, which was written correctly and then not
        # copied. A namespace package or a broken install gives no origin, and crashing here
        # would report a checker bug where the install is what is wrong.
        check("the package has a resolvable location", False, "no spec origin for senbonzakura")
        return 1
    pkg = _pl.Path(spec2.origin).parent
    shipped = sorted((pkg / "vendor" / "bin").glob("*/llama-quantize"))
    verdict = next((ln for ln in out.splitlines() if "llama-quantize" in ln), "")
    runs = verdict.strip().startswith("\u2713")

    # WHOSE FAULT IT IS, AND THIS CHECK USED TO BE CONFIDENTLY WRONG ABOUT IT.
    #
    # It reported every non-starting binary as "a packaging fault". The commonest reason a
    # vendored `llama-quantize` will not start in here is not packaging at all: llama.cpp links
    # OpenMP, `python:3.13-slim` does not ship `libgomp1`, and this project's own Dockerfile
    # installs it explicitly with a comment saying why. The binary is exactly what we think it is
    # and the image is missing a dependency of it, which `doctor` already says correctly and this
    # file then contradicted.
    #
    # BEING WRONG ABOUT THE CAUSE IS NOT A SMALL THING HERE: a confidently misattributed gate
    # sends somebody to re-vendor a correct binary, which is the exact failure mode this project
    # keeps finding in other people's tooling and had shipped in its own.
    #
    # NOR IS IT A CLEAN BILL OF HEALTH. A user who pip-installs the platform wheel onto a slim
    # base meets the same wall, and a wheel cannot declare a system package. So it still fails,
    # with the honest name: an undeclared system dependency.
    missing_library = blames_the_image(verdict)

    if shipped and not runs and missing_library:
        check("a shipped quantiser actually starts", False,
              f"the wheel installed {shipped[0]} and it cannot start because this image lacks a "
              f"library it links: {verdict.strip()!r}. The binary is correct. This is an "
              f"UNDECLARED SYSTEM DEPENDENCY, which a wheel has no way to express, so anybody "
              f"installing onto a base image this slim meets it too.")
        print("\n  note  the binaries this wheel carries are correct and this IMAGE cannot run "
              "them: it is missing a shared library they link, which on Debian and Ubuntu is "
              "usually libgomp1. Re-vendoring will not help. Install the library, or state the "
              "requirement where somebody installing the wheel will read it.\n")
    elif shipped and not runs:
        check("a shipped quantiser actually starts", False,
              f"the wheel installed {shipped[0]} and doctor rejects it: {verdict.strip()!r}. "
              f"A wheel that carries a capability it cannot deliver is worse than one that "
              f"carries neither, because only the second is honest about it.")
        print("\n  note  this wheel CARRIES binaries that do not run here, and doctor does not "
              "name a missing library, so the binaries themselves are suspect. That is a "
              "packaging fault, not a missing feature.\n")
    elif shipped:
        print("\n  note  this is a PLATFORM wheel and its binaries run.\n")
        check("a platform wheel can convert and quantise", "\u2717" not in verdict, verdict)
    else:
        print("\n  note  this is a UNIVERSAL wheel: it carries no binaries, so doctor should "
              "refuse.\n")
        # Exiting 0 while printing "this install cannot do what it claims" is the failure that
        # made this whole file necessary.
        check("doctor exits non-zero when it cannot do the job", r.returncode != 0,
              f"exit {r.returncode}")
        check("doctor says plainly that the install is not usable",
              "cannot do what it claims" in out, out[-300:])

    # ── the corpora, and whether this wheel is meant to carry them ───────────────
    #
    # THE CHECK THAT WAS A LABEL RATHER THAN A RESULT. This asked whether the string
    # "corpus advbench" appeared in doctor's output, and doctor prints that string on BOTH
    # outcomes: "✓ corpus advbench  520 prompts" and "✗ corpus advbench  the bundled corpora
    # are not installed". So it passed on a wheel containing no corpora at all, which is
    # precisely the presence-versus-works failure this file's own docstring exists to prevent.
    # Only the neighbouring row-count check ever caught it.
    #
    # And a wheel built from a plain CLONE genuinely has no corpora: `corpora.bin` and
    # `default-track.bin` are generated release artefacts that are deliberately not in git,
    # because they hold harmful prompts and the corpus is published as a gated dataset. So
    # there are two legitimate wheels, and the check now says which one it is looking at
    # rather than failing the honest case or passing the broken one.
    r = run(["doctor"])
    out = r.stdout + r.stderr
    carries_corpora = "\u2713  corpus advbench" in out or "\u2713 corpus advbench" in out
    if carries_corpora:
        check("bundled corpora are present in the wheel AND load", True, "doctor reports a pass")
        try:
            from senbonzakura import corpora
            rows = corpora.load("advbench")
            check("a bundled corpus loads offline", len(rows) == 520, f"{len(rows)} rows")
        except Exception as e:
            check("a bundled corpus loads offline", False, f"{type(e).__name__}: {e}")
    else:
        note("this wheel carries NO bundled corpora, so it is a code-only build. A RELEASE "
             "wheel must carry them: run `senbonzakura corpora` and tools/packaging/pack_track.py first.")
        # The property that has to hold for a code-only wheel is that the absence is legible.
        try:
            from senbonzakura import corpora
            corpora.load("advbench")
            check("a wheel without corpora refuses rather than pretending", False,
                  "it returned rows from somewhere")
        except Exception as e:
            said = str(e)
            check("a wheel without corpora refuses in words that name the builder",
                  "senbonzakura corpora" in said, said[:200])

    # ── the refusals are clean, not tracebacks ───────────────────────────────────
    # True of both wheel kinds: a missing INPUT file must be refused with a sentence rather than
    # a traceback, whether or not the binary that would have processed it is present.
    for cmd, needle in (("quantise", "llama-quantize"), ("convert", "convert")):
        r = run([cmd, "--help"])
        check(f"{cmd} --help works without the binary", r.returncode == 0,
              f"exit {r.returncode}")
        # An output path inside a disposable container with a read-only root: /tmp is the only
        # writable place and nothing here is shared with a host.
        out_path = "/tmp/out.gguf"  # noqa: S108
        r = run([cmd, "/nonexistent.gguf", out_path]
                + (["--type", "Q4_K_M"] if cmd == "quantise" else []))
        check(f"{cmd} refuses rather than crashing", r.returncode != 0, f"exit {r.returncode}")
        check(f"{cmd} refuses without a traceback",
              "Traceback (most recent call last)" not in (r.stdout + r.stderr),
              (r.stdout + r.stderr)[-300:])
        del needle

    # ── a returned failure code survives to the shell ────────────────────────────
    # `python -m` used to discard these, so a caller checking the status saw success.
    r = subprocess.run([sys.executable, "-m", "senbonzakura", "interactive"],
                       capture_output=True, text=True, input="", timeout=120, check=False)
    check("a non-terminal refusal exits non-zero", r.returncode != 0, f"exit {r.returncode}")

    # ── nothing reached out ──────────────────────────────────────────────────────
    check("no credentials are visible to this process",
          not any(k in __import__("os").environ for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
                                                          "AWS_SECRET_ACCESS_KEY", "SSH_AUTH_SOCK")))

    print(f"\n{CHECKS} checks, {CHECKS - len(FAILURES)} pass, {len(FAILURES)} failed")
    if FAILURES:
        print("\nfailed: " + ", ".join(FAILURES))
        print(json.dumps({"checks": CHECKS, "failed": FAILURES}))
        return 1
    print("\nThis install behaves the way a fresh install should, refusals included.")
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
