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
import re
import shutil
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
    try:
        p = subprocess.run([sys.executable, "-m", *argv], capture_output=True, text=True,
                           timeout=timeout, check=False, cwd=str(ROOT))
    except subprocess.TimeoutExpired as e:
        # Without this the one case TIMEOUT exists for gets a raw traceback instead of this
        # file's own diagnosis, which is the worst message for the worst failure.
        tail = (e.output or b"")[-2000:] if isinstance(e.output, bytes) else (e.output or "")[-2000:]
        raise SmokeError(f"{name}: still running after {timeout}s and was killed. A smoke job that "
                         f"hangs burns a runner and tells nobody anything.\n{tail}") from None
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


#: Strings the guided mode's FIRST screen prints before it asks anything. Content, not volume: the
#: first version of this stage asserted only that some bytes arrived and that they were not the
#: no-terminal refusal, and a Python traceback satisfies both. See `drive_the_guided_mode`.
GUIDED_FIRST_SCREEN = ("Senbonzakura, guided mode.", "What would you like to do?")

#: The read loop stops when it sees this, because it is the point past which the mode is waiting
#: for a keystroke. Stopping on a bare ">" instead meant stopping on `File "<frozen runpy>"`.
GUIDED_SENTINEL = "What would you like to do?"

#: How long the guided mode must still be alive after asking, before we believe it is waiting for
#: an answer rather than having printed the screen on its way out.
GUIDED_MUST_WAIT_FOR = 3.0


def drive_the_guided_mode():
    """Start the guided mode on a REAL terminal and check it asks its first question.

    The unit tests fake `isatty`, which is why the guided mode once shipped 48 passing tests
    having never run: a fake terminal cannot catch a real one. This allocates a pty, so the code
    takes the branch a person takes.

    WHAT THIS ASSERTS, AND WHY IT IS NOT "SOME BYTES ARRIVED"

    The 2026-09-10 panel proved by mutation that the first version of this stage passed against a
    guided mode that crashed on every real terminal. It broke its read loop on `"?" in seen or ">"
    in seen` and then asserted only that the output was non-empty and was not the no-terminal
    refusal. A Python traceback contains `>` (`File "<frozen runpy>"`, `in <module>`), so a crash
    satisfied the break condition and failed neither assertion.

    Two mutations were run. Raising unconditionally in `interactive.run` was caught by the unit
    tests, so the suite was not blind. Raising only when `sys.stdin.isatty()` is genuinely true is
    the mutant the unit tests structurally cannot see, and it passed: seventy unit tests green and
    a stage reporting `ok, it rendered 1654 bytes on a terminal`.

    So this now asserts on what the first screen says, refuses a traceback explicitly, and checks
    the child was still alive when the read ended. It stays deliberately shallow past that point:
    walking the whole menu in CI is a keystroke-ordering test that breaks on every rewording, and
    the wording assertions belong in the unit tests.
    """
    import pty
    import select
    import signal

    print("\n=== interactive, on a real pty ===", flush=True)
    pid, fd = pty.fork()
    if pid == 0:                                   # pragma: no cover - the child is replaced
        os.chdir(str(ROOT))
        try:
            os.execv(sys.executable,  # noqa: S606 - fixed argv, no shell, in a pty child
                     [sys.executable, "-m", "senbonzakura", "interactive"])
        finally:
            # execv only returns if it FAILED. Without this the child falls through into the
            # parent's code below and reads a file descriptor that means nothing to it.
            os._exit(127)

    seen, deadline = "", time.time() + 90
    try:
        while time.time() < deadline and len(seen) < 8000:
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
            if GUIDED_SENTINEL in seen:
                break
        # Still waiting for a keystroke, rather than having exited while we were reading? A
        # process that printed the right screen and then died is not a working guided mode.
        #
        # The grace period is load-bearing. Asking immediately after the read loop breaks is a
        # race: the mutant that printed the question and then exited was still alive at that
        # instant, so a bare WNOHANG call reported it as waiting and the stage passed. Give it time
        # to actually be gone before concluding it is not.
        alive, until = True, time.time() + GUIDED_MUST_WAIT_FOR
        while time.time() < until:
            if os.waitpid(pid, os.WNOHANG) != (0, 0):
                alive = False
                break
            time.sleep(0.1)
    finally:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        os.close(fd)
        try:
            os.waitpid(pid, 0)
        except (ChildProcessError, OSError):
            pass

    if "Traceback (most recent call last)" in seen:
        raise SmokeError("interactive CRASHED on a real terminal. The unit tests fake isatty, so "
                         f"they cannot see this:\n{seen[-2000:]}")
    if not seen.strip():
        raise SmokeError("interactive on a pty printed nothing at all. It detects a terminal with "
                         "isatty, so on a real one it should render its first question.")
    if "needs a terminal" in seen:
        raise SmokeError(f"interactive was given a real pty and still said it needs a terminal:"
                         f"\n{seen[:600]}")
    for needle in GUIDED_FIRST_SCREEN:
        if needle not in seen:
            raise SmokeError(f"interactive rendered {len(seen)} bytes on a terminal without ever "
                             f"printing {needle!r}, so whatever it did was not asking its first "
                             f"question.\n{seen[-2000:]}")
    if not alive:
        raise SmokeError("interactive printed its first screen and then exited instead of waiting "
                         f"for an answer.\n{seen[-2000:]}")
    print(f"  ok, it asked its first question on a terminal and waited ({len(seen)} bytes)",
          flush=True)


#: How far the documented compass AUC may sit from the one the tool prints. Not zero: the figure
#: moves with the transformers version (0.9653 on 5.14.1, 0.9861 on 5.13.1) and the page says so.
#: Wide enough to survive a minor dependency bump, narrow enough that the 1.0000 the page printed
#: for months, and the 0.9826 it printed after that, would both have failed here.
COMPASS_TOLERANCE = 0.02
COMPASS_DOC = ROOT / "docs" / "guide" / "compass.md"


def documented_compass_auc(path=COMPASS_DOC):
    """The AUC the compass page tells the reader they will see."""
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^MARGIN_DONE\s+auc=([0-9.]+)", text, re.MULTILINE)
    if not m:
        raise SmokeError(f"{path} no longer contains a MARGIN_DONE line, so the worked example "
                         f"this stage checks has moved or gone. Update the stage or the page.")
    return float(m.group(1))


def check_the_documented_compass_figure(out):
    """Run the command the compass page prints, and compare what comes back to what it promises.

    A review pass on 2026-09-10 ran that command verbatim and got 0.9653 where the page said
    0.9826 and claimed "measured twice, byte-identical". Both readings were real; neither said
    which environment produced it, and the difference was the transformers version. The page had
    already carried a warning box about exactly this having happened once before, and nothing
    checked it, because `tests/test_documented_figures.py` deliberately pins the prose against the
    numbers rather than the numbers against the tool.

    The smoke already downloads this model and runs on CPU, so closing that gap costs one command.
    """
    want = documented_compass_auc()
    written = out / "compass-doc.json"
    text = run("the compass figure the docs promise",
               ["senbonzakura", "compass", "--model", MODEL,
                "--harmful", str(TRACK / "bad_eval_ds"), "--harmless", str(TRACK / "good_ds"),
                "--skip-harmful", "0", "--skip-harmless", "0", "--n", "12",
                "--out", str(written), "--device", "cpu"],
               expect_marker="MARGIN_DONE")
    m = re.search(r"MARGIN_DONE\s+auc=([0-9.]+)", text)
    if not m:
        raise SmokeError("compass printed MARGIN_DONE with no auc= on it")
    got = float(m.group(1))
    if abs(got - want) > COMPASS_TOLERANCE:
        raise SmokeError(
            f"docs/guide/compass.md promises the reader auc={want:.4f} and running the command it "
            f"prints gives auc={got:.4f}, a gap of {abs(got - want):.4f} against a tolerance of "
            f"{COMPASS_TOLERANCE}. A worked example that does not reproduce teaches a reader to "
            f"distrust the next one. Re-run the block and update the page, or the environment "
            f"moved and the page's note about which one it was measured on needs updating too.")
    print(f"  ok, the page promises {want:.4f} and the tool gives {got:.4f}")


#: Stamped into every JSON the smoke writes. The toy track is 8 harmful / 4 eval / 8 harmless
#: rows and its harmful rows are placeholders, so baseline refusal is 0.0 and refusal REMOVAL
#: cannot be demonstrated on it at all. The run proves the pipeline executes and writes
#: artefacts; every NUMBER in those artefacts is meaningless.
#:
#: Without this the files are shaped exactly like a real run's, carry a real `provenance` block,
#: and would be quoted by anyone who found one. This project has already published a p-value
#: taken from its own synthetic fixture; that was caught by a reviewer rather than by anything
#: in the tree, and this is the thing in the tree.
SMOKE_NOTICE = (
    "Produced by tools/smoke_end_to_end.py on examples/toy-track, which holds 8 harmful, "
    "4 harmful-eval and 8 harmless rows, and whose harmful rows are placeholders. This file "
    "proves the pipeline ran and wrote an artefact. Every number in it is meaningless and must "
    "not be quoted, compared, or published."
)


#: OUR result artefacts, by name. NOT a glob over `*.json`, and the difference is not tidiness:
#: the saved model directory holds `config.json`, `tokenizer.json`, `tokenizer_config.json` and
#: `generation_config.json`, which belong to transformers and are read by every loader. The first
#: version of this swept every JSON under the output directory and stamped all four of them,
#: which is editing a model's configuration to leave a note in it.
SMOKE_ARTEFACTS = frozenset({
    "score.json", "compass.json", "compass-doc.json", "drift.json",
    "abliteration.json", "run.json", "best-config.json", "trials.json",
})


def stamp_smoke_artefacts(out):
    """Mark the result artefacts the smoke produced. Returns how many it touched.

    A list-shaped artefact (`trials.json`) has nowhere to put a key, so it is wrapped in an
    object that carries the notice and keeps the rows under `trials`. That changes its shape,
    which is acceptable HERE and only here: these files exist for about a minute inside a smoke
    run and nothing reads them afterwards. It would not be acceptable for a real run's output.
    """
    stamped = []
    for path in sorted(out.rglob("*.json")):
        if path.name not in SMOKE_ARTEFACTS:
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        doc = ({"smoke_artefact": SMOKE_NOTICE, "trials": doc} if isinstance(doc, list)
               else {**doc, "smoke_artefact": SMOKE_NOTICE} if isinstance(doc, dict) else None)
        if doc is None:
            continue
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        stamped.append(path.name)
    return stamped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()) / "smoke")
    ap.add_argument("--skip-interactive", action="store_true",
                    help="skip the pty stage, for a platform with no pty")
    a = ap.parse_args(argv)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out / "model"
    # CI gets a fresh RUNNER_TEMP, but the docstring invites a person to run this by hand, and the
    # second run then trips the abliterator's output pre-flight on an occupied directory.
    shutil.rmtree(model_dir, ignore_errors=True)
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

    # THE ASSERTION THAT MAKES THIS STAGE MEAN ITS OWN TITLE. Checking the keys are non-null
    # distinguishes "abliterated" from "crashed", not from "wrote a file with a number in it": a
    # bake applying a zero-magnitude edit writes both and passes.
    #
    # WHAT THIS CANNOT ASSERT, AND WHY. The obvious check is that refusal fell. It cannot be made
    # here: the toy track's harmful rows are synthetic placeholders ("example harmful request
    # number 0..."), so `baseline_refusals` is already 0.0 and there is nothing to remove. This
    # smoke proves the machinery runs end to end and edits real weights; it does NOT demonstrate
    # refusal removal, and saying otherwise would be exactly the kind of green this file exists to
    # refuse. Demonstrating that needs a real track and a real model, which is a GPU job.
    #
    # What does discriminate, on any track: an edit that changed nothing leaves the output
    # distribution identical to the base, so the measured KL is exactly zero. A non-zero KL is the
    # model itself reporting that its weights moved.
    kl, edits = doc["post_bake_kl"], sum(doc.get("directions_per_layer") or [])
    if not kl > 0:
        raise SmokeError(
            f"the bake changed nothing: post_bake_kl is {kl}, so the edited model's output "
            f"distribution is identical to the base model's. This job is titled 'it can actually "
            f"abliterate a model', and a zero-magnitude edit has not done that however cleanly it "
            f"exited.")
    if not edits > 0:
        raise SmokeError(f"no layer received a direction ({doc.get('directions_per_layer')}), so "
                         f"nothing was ablated anywhere.")
    baseline = doc.get("baseline_refusals")
    if baseline is None:
        raise SmokeError("abliteration.json carries no baseline_refusals, so nothing records what "
                         "the model did before the edit")
    if baseline > 0 and not doc["post_bake_refusals"] < baseline:
        raise SmokeError(f"the track had refusals to remove ({baseline}) and the edit removed none "
                         f"({doc['post_bake_refusals']})")
    print(f"  the edit reached {edits} layers and moved the model: kl={kl:.4f} "
          f"(refusal {baseline} -> {doc['post_bake_refusals']}; the toy track has no real refusals "
          f"to remove, so this stage proves the machinery, not the result)")

    run("score the model that was just made",
        ["senbonzakura", "score", "--model", str(model_dir),
         "--eval", str(TRACK / "bad_eval_ds"), "--out", str(out / "score.json"),
         "--device", "cpu", "--n", "8", "--skip", "0", "--max-new", "16", "--batch", "4"],
        expect_marker="SCORE_DONE")

    check_the_documented_compass_figure(out)

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

    # AFTER everything, so nothing the smoke writes can leave here looking like a measurement.
    stamped = stamp_smoke_artefacts(out)
    # The model's own configuration must come through untouched. A stamp that edits
    # `config.json` is not a note, it is a change to what the loader reads.
    for untouchable in ("config.json", "tokenizer.json", "tokenizer_config.json",
                        "generation_config.json"):
        for found in out.rglob(untouchable):
            doc = json.loads(found.read_text(encoding="utf-8"))
            if isinstance(doc, dict) and "smoke_artefact" in doc:
                raise SmokeError(
                    f"{found} carries the smoke notice, and it is a file transformers reads. "
                    f"The stamp must name our artefacts rather than sweep every JSON.")
    if not stamped:
        raise SmokeError(
            "no JSON artefacts were found to stamp, which means either the run wrote none (so "
            "the smoke proved less than it claims) or they went somewhere this did not look. "
            "Either way the notice is not on them.")
    print(f"\n=== every artefact marked as a smoke artefact ===\n  stamped {len(stamped)}: "
          f"{', '.join(stamped)}")

    print(f"\nSMOKE_OK: abliterated, measured and reported in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":      # pragma: no cover
    try:
        raise SystemExit(main())
    except SmokeError as e:
        print(f"\nSMOKE_FAILED: {e}", file=sys.stderr)
        raise SystemExit(1) from None
