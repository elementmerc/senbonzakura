#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
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

    # ── it is genuinely a fresh install ──────────────────────────────────────────
    spec = importlib.util.find_spec("senbonzakura")
    origin = spec.origin if spec else ""
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

    # ── doctor tells the truth about a fresh install ─────────────────────────────
    # The binaries are not in the wheel. doctor must say so and must exit non-zero, because
    # exiting 0 while printing "this install cannot do what it claims" is the failure that
    # made this whole file necessary.
    check("doctor reports the missing quantiser", "llama-quantize" in out)
    check("doctor exits non-zero when it cannot do the job", r.returncode != 0,
          f"exit {r.returncode}")
    check("doctor says plainly that the install is not usable",
          "cannot do what it claims" in out, out[-300:])

    # ── the corpora DO ship, and work with no network ────────────────────────────
    r = run(["doctor"])
    check("bundled corpora are present in the wheel", "corpus advbench" in (r.stdout + r.stderr))
    try:
        from senbonzakura import corpora
        rows = corpora.load("advbench")
        check("a bundled corpus loads offline", len(rows) == 520, f"{len(rows)} rows")
    except Exception as e:
        check("a bundled corpus loads offline", False, f"{type(e).__name__}: {e}")

    # ── the refusals are clean, not tracebacks ───────────────────────────────────
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
