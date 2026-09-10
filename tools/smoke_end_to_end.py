#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Abliterate a real model, measure it, and drive the guided mode. On CPU, in about a minute.

WHY THIS EXISTS

CI ran 3048 unit tests on three operating systems and three interpreters and had **never
abliterated a model**. Every runner is CPU-only, so the thing this tool is for was exercised by
nothing except hand-run jobs on one laptop. A suite that green-lights every commit without once
doing the job is a suite measuring the parts rather than the product.

It turns out not to need a GPU. SmolLM2-135M-Instruct plus the toy track committed in
`examples/toy-track` runs a genuine search, bakes the winner, saves a real model with provenance,
and takes 25 seconds on a CPU. Everything downstream of it takes another 20.

WHAT IT ASSERTS, AND WHY ON OUTPUT RATHER THAN EXIT CODES

On 2026-08-05 five benchmark jobs reported success having measured nothing, because the scripts
printed a success line unconditionally. So every stage here is checked on a marker the code emits
only when it actually did the work: `DONE`, `SCORE_DONE`, `MARGIN_DONE`, `DRIFT_DONE`. An exit code
is necessary and it is not sufficient, and this file is the place that distinction is cheapest to
honour.

Three stages assert a REFUSAL rather than a success, because a tool that cannot say no is a tool
whose approvals mean nothing:

  kageyoshi on a track too small   its auto budget wants 64 eval rows and the toy track has 4,
                                   so it must refuse rather than score the published partition
  interactive without a terminal   it must say so and point at the flags
  interactive WITH a terminal      driven through a real pty, because the unit tests fake isatty
                                   and a fake tty has never caught a real one
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
TRACK = ROOT / "examples" / "toy-track"

#: Every stage is bounded. A smoke job that hangs is worse than one that fails, because it burns a
#: runner for six hours and tells nobody anything.
TIMEOUT = 900


class SmokeError(Exception):
    pass


def run(name, argv, *, expect_marker=None, expect_text=None, expect_fail=False, timeout=TIMEOUT):
    """One stage. Reports what it looked for, not merely that something happened."""
    started = time.time()
    print(f"\n=== {name} ===", flush=True)
    p = subprocess.run([sys.executable, "-m", *argv], capture_output=True, text=True,
                       timeout=timeout, check=False, cwd=str(ROOT))
    out = p.stdout + p.stderr
    took = time.time() - started

    if expect_fail:
        if p.returncode == 0:
            raise SmokeError(f"{name}: expected a refusal and it exited 0.\n{out[-2000:]}")
    elif p.returncode != 0:
        raise SmokeError(f"{name}: exited {p.returncode}.\n{out[-2000:]}")

    for needle in filter(None, [expect_marker, expect_text]):
        if needle not in out:
            raise SmokeError(f"{name}: exited {p.returncode} without ever printing {needle!r}, so "
                         f"nothing here shows the work was done.\n{out[-2000:]}")

    shown = expect_marker or expect_text or "(exit status only)"
    print(f"  ok in {took:.0f}s, saw {shown!r}", flush=True)
    return out


def drive_the_guided_mode():
    """Start the guided mode on a REAL terminal and check it asks its first question.

    The unit tests fake `isatty`, which is why the guided mode once shipped 48 passing tests
    having never run: a fake terminal cannot catch a real one. This allocates a pty, so the code
    takes the branch a person takes.

    Deliberately shallow. Walking the whole menu in CI is a keystroke-ordering test that breaks
    every time a question is reworded, and what is worth protecting here is that the mode starts,
    detects the terminal, and renders. The wording assertions live in the unit tests.
    """
    import pty
    import select
    import signal

    print("\n=== interactive, on a real pty ===", flush=True)
    pid, fd = pty.fork()
    if pid == 0:                                   # pragma: no cover - the child is replaced
        os.chdir(str(ROOT))
        os.execv(sys.executable,  # noqa: S606 - fixed argv, no shell, in a pty child
                 [sys.executable, "-m", "senbonzakura", "interactive"])

    seen, deadline = "", time.time() + 90
    try:
        while time.time() < deadline and len(seen) < 4000:
            r, _, _ = select.select([fd], [], [], 5)
            if not r:
                break
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            seen += chunk.decode("utf-8", "replace")
            if "?" in seen or ">" in seen:
                break
    finally:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        os.close(fd)
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass

    if not seen.strip():
        raise SmokeError("interactive on a pty printed nothing at all. It detects a terminal with "
                     "isatty, so on a real one it should render its first question.")
    if "needs a terminal" in seen:
        raise SmokeError(f"interactive was given a real pty and still said it needs a terminal:\n{seen[:600]}")
    print(f"  ok, it rendered {len(seen)} bytes on a terminal", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()) / "smoke")
    ap.add_argument("--skip-interactive", action="store_true",
                    help="skip the pty stage, for a platform with no pty")
    a = ap.parse_args(argv)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out / "model"
    started = time.time()

    # Reports honestly on a source checkout, where the release-time artefacts are absent. Its
    # exit code is EXPECTED to be non-zero here, so what is checked is that it says why.
    run("doctor tells the truth about an incomplete install",
        ["senbonzakura", "doctor"], expect_text="checks,", expect_fail=True)

    run("setup reads this machine",
        ["senbonzakura", "setup"], expect_text="platform")

    # THE STAGE THIS FILE EXISTS FOR: a real search, a real bake, a real model on disk.
    run("abliterate a real model, end to end",
        ["senbonzakura", "abliterate", "--model", MODEL, "--track", str(TRACK),
         "--out", str(model_dir), "--device", "cpu", "--trials", "2", "--dir-prompts", "8",
         "--eval-refusal", "4", "--eval-kl", "4", "--eval-refusal-final", "4",
         "--gen-tokens", "16", "--gen-batch", "4", "--no-persist-study"],
        expect_marker="DONE")

    for want in ("config.json", "abliteration.json", "run.json"):
        if not (model_dir / want).is_file():
            raise SmokeError(f"the run reported DONE and wrote no {want}")
    doc = json.loads((model_dir / "abliteration.json").read_text(encoding="utf-8"))
    for key in ("post_bake_refusals", "post_bake_kl"):
        if doc.get(key) is None:
            raise SmokeError(f"abliteration.json carries no {key}, so the run measured nothing")
    print(f"  the edited model records refusals={doc['post_bake_refusals']} kl={doc['post_bake_kl']:.4f}")

    run("score the model that was just made",
        ["senbonzakura", "score", "--model", str(model_dir),
         "--eval", str(TRACK / "bad_eval_ds"), "--out", str(out / "score.json"),
         "--device", "cpu", "--n", "8", "--skip", "0", "--max-new", "16", "--batch", "4"],
        expect_marker="SCORE_DONE")

    run("the compass, on the same model",
        ["senbonzakura", "compass", "--model", str(model_dir),
         "--harmful", str(TRACK / "bad_eval_ds"), "--harmless", str(TRACK / "good_ds"),
         "--skip-harmful", "0", "--skip-harmless", "0", "--n", "8",
         "--out", str(out / "compass.json"), "--device", "cpu"],
        expect_marker="MARGIN_DONE")

    prompts = out / "prompts.txt"
    sys.path.insert(0, str(ROOT / "src"))
    from senbonzakura.headtohead_stage import load_texts
    prompts.write_text("\n".join(load_texts(str(TRACK / "good_ds"), 8)) + "\n", encoding="utf-8")

    run("drift, against the base it was made from",
        ["senbonzakura", "drift", "--model", str(model_dir), "--base", MODEL,
         "--prompts", str(prompts), "--out", str(out / "drift.json"), "--label", "smoke",
         "--batch", "4"],
        expect_marker="DRIFT_DONE")

    run("the model card that would travel with it",
        ["senbonzakura", "report", "--abliteration", str(model_dir / "abliteration.json"),
         "--out", str(out / "CARD.md"), "--base-licence", "apache-2.0"],
        expect_text="wrote")
    card = (out / "CARD.md").read_text(encoding="utf-8")
    for want in ("license: apache-2.0", "base model's licence"):
        if want not in card:
            raise SmokeError(f"the model card does not carry {want!r}")

    # ── the refusals, which are the half a smoke test usually forgets ────────────────
    run("kageyoshi refuses a track too small for its own budget",
        ["senbonzakura", "kageyoshi", "--model", MODEL, "--track", str(TRACK),
         "--out", str(out / "never"), "--device", "cpu", "--no-persist-study"],
        expect_text="rows this track holds for selection", expect_fail=True)

    run("interactive refuses a pipe and names the alternative",
        ["senbonzakura", "interactive"], expect_text="needs a terminal", expect_fail=True)

    if not a.skip_interactive and hasattr(os, "openpty"):
        drive_the_guided_mode()
    else:
        print("\n=== interactive, on a real pty ===\n  skipped: no pty on this platform")

    print(f"\nSMOKE_OK: abliterated, measured and reported in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":      # pragma: no cover
    try:
        raise SystemExit(main())
    except SmokeError as e:
        print(f"\nSMOKE_FAILED: {e}", file=sys.stderr)
        raise SystemExit(1) from None
