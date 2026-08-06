"""Run a head-to-head between abliteration tools, on one machine, in one command.

`senbonzakura bench head-to-head --tools senbon,heretic --seeds 5` is the whole thing: it runs
each tool's arms, scores every model it produced with one judge, and writes a report. No
orchestrator, no ssh, no private tooling. A stranger can run it, which is the point.

**Why this is a command and not a run spec** (decision Q-5, 2026-08-06). This benchmark used to
exist as four hundred lines of orchestrator configuration with shell scripts embedded in it, and
most of the defects found on 2026-08-05 and 06 were in that shell rather than in either tool: two
resume guards that asked for keys the artefacts never carried, a success marker printed whether or
not the work happened, a scorer invoked three wrong ways at once. Configuration cannot be tested
and does not ship in a wheel. A head-to-head is also not an orchestration problem: it is run tool
A n times, run tool B n times, score everything with one judge, report. That is a loop.

What an orchestrator is genuinely for stays with the orchestrator: which machine, retries across a
reboot, arbitrating a GPU between jobs, streaming a running log somewhere durable. Its spec now
holds one line, which is this command.

**Isolation is a flag, not an architecture.** `--isolate docker` runs each arm sealed: no network,
read-only inputs, no credentials, dropped capabilities. It is off by default because a tool that
cannot run without docker cannot be run by most people who want to check our numbers, and it warns
rather than proceeding quietly when asked for and unavailable. Running somebody else's abliteration
tool on your own machine deserves a sandbox, and that matters more to a stranger than to us.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Every arm writes this when it has finished, and the runner checks the file rather than the word.
# A marker in a log is a string, and anything that can print it can claim the work: that is how a
# rehearsal once reported five jobs done having measured nothing.
ARM_MANIFEST = "arm.json"

# The prompt files every tool is scored on, staged once so no tool brings its own.
SLICE_FILES = ("good.txt", "bad.txt", "keyword_prompts.txt", "final_prompts.txt",
               "kl_prompts.txt")
SLICE_PROVENANCE = "slices.json"


class BenchError(RuntimeError):
    """A failure the operator can act on, phrased for a person rather than a stack trace."""


@dataclass(frozen=True)
class Adapter:
    """How to run one tool, and how to tell afterwards whether it really ran.

    Adding a rival tool is adding one of these. It carries no logic, so it can be contributed by
    somebody who did not write this file and reviewed as data rather than as code.
    """

    name: str
    # Built as a list, never a string: a shell string is how a path with a space becomes two
    # arguments and how a corpus name becomes a command.
    argv: object
    # What the tool leaves behind that proves it ran, relative to the arm directory.
    produces: tuple[str, ...]
    # Where the finished model ends up, relative to the arm directory. The two tools disagree:
    # senbonzakura writes straight into its output directory, so the arm directory IS the model,
    # while Heretic's comes from a selection pass in a subdirectory. Getting this wrong scored one
    # tool's arms against nothing at all on 2026-08-05.
    model_subdir: str = ""
    # Read the tool's own reported figures out of its artefacts, as {key: value}. Never compared
    # across tools: two tools' KL figures come from different estimators on different slices, and
    # putting them in one column is the error four claims were withdrawn for.
    self_report: object = None
    notes: str = ""


def _senbon_argv(*, model, track, out, seed, trials, slices, extra):
    # `--resume` is not optional: a retried arm otherwise meets the study its own first attempt
    # created and dies on a duplicated-study error, which is a crash on the happy path.
    return ["python", "-u", "-m", "senbonzakura", "kageyoshi",
            "--model", str(model), "--track", str(track), "--out", str(out),
            "--seed", str(seed), "--trials", str(trials),
            "--patience", "0", "--device", "cuda", "--resume", *extra]


def _senbon_report(arm: Path):
    doc = _read_json(arm / "abliteration.json")
    return {
        "refusals": doc.get("post_bake_refusals"),
        "kl": doc.get("post_bake_kl"),
        "trials_ran": doc.get("trials_ran"),
        "refusal_estimator": "senbonzakura rulers, our search eval",
        "kl_estimator": "senbonzakura, our coherence slice",
    }


def _heretic_argv(*, model, track, out, seed, trials, slices, extra):
    # Heretic owns two scorers and both default to fetching their prompts from the Hub, which a
    # sealed box cannot do. More importantly, a tool's search is steered by whatever its scorers
    # measure, so scoring the two tools on different prompts is not giving them the same problem.
    # The slices are staged once, by us, and both tools read them.
    if slices is None:
        raise BenchError(
            "heretic needs the staged evaluation slices; pass --eval-slices. They are what make "
            "both tools read the same prompts, and without them Heretic fetches its own")
    s = Path(slices)
    return ["python", "-u", "-m", "bench.run_heretic",
            "--model", str(model), "--out", str(out),
            "--good", str(s / "good.txt"), "--bad", str(s / "bad.txt"),
            "--keyword-prompts", str(s / "keyword_prompts.txt"),
            "--kl-prompts", str(s / "kl_prompts.txt"),
            "--seed", str(seed), "--trials", str(trials), *extra]


def _heretic_report(arm: Path):
    doc = _read_json(arm / "best_of_n.json")
    winner = doc.get("winner") or {}
    return {
        "refusals": winner.get("refusals"),
        "kl": winner.get("kl"),
        "trials_ran": doc.get("trials_ran"),
        "refusal_estimator": "Heretic, its own keyword scorer",
        "kl_estimator": "Heretic, its own evaluation",
    }


ADAPTERS: dict[str, Adapter] = {
    "senbon": Adapter(
        name="senbon",
        argv=_senbon_argv,
        produces=("abliteration.json", "config.json"),
        model_subdir="",
        self_report=_senbon_report,
        notes="this tool; writes the model straight into the arm directory",
    ),
    "heretic": Adapter(
        name="heretic",
        argv=_heretic_argv,
        produces=("best_of_n.json",),
        model_subdir="model",
        self_report=_heretic_report,
        notes="p-e-w/heretic, run through the equal-budget selection pass",
    ),
}


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        raise BenchError(f"{path} exists but could not be read: {e}") from e


# ── the resume guard ──────────────────────────────────────────────────────────────────
def arm_manifest(adapter: Adapter, seed: int, model: str, trials: int) -> dict:
    """What an arm must agree with to count as already done.

    Written by the arm on success and compared on the next run. It names the configuration rather
    than merely existing, because "a file is here" is what a guard says when it has stopped
    guarding: on 2026-08-04 a run reported seven jobs complete while every one of them read
    yesterday's output, and both guards written to prevent a repeat then asked their artefacts for
    a key those artefacts had never carried, so neither could ever fire.
    """
    return {"tool": adapter.name, "seed": int(seed), "model": str(model), "trials": int(trials)}


def arm_is_done(arm: Path, expected: dict, adapter: Adapter) -> tuple[bool, str]:
    """Has this exact arm already been produced? Returns the verdict and why, always."""
    arm = Path(arm)
    manifest = arm / ARM_MANIFEST
    if not manifest.is_file():
        return False, "no manifest, so nothing here claims to be finished"
    got = _read_json(manifest)
    for key, want in expected.items():
        if str(got.get(key)) != str(want):
            return False, f"manifest says {key}={got.get(key)!r}, this run wants {want!r}"
    missing = [p for p in adapter.produces if not (arm / p).exists()]
    if missing:
        return False, f"manifest agrees but {', '.join(missing)} is missing"
    return True, "manifest agrees and every declared artefact is present"


def write_arm_manifest(arm: Path, expected: dict) -> Path:
    """Written LAST, and only after the artefacts are checked, so it cannot vouch for a partial arm."""
    path = Path(arm) / ARM_MANIFEST
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def slices_match_track(slices: Path, track: Path) -> list[str]:
    """Were these prompt files cut from this corpus?

    Once one tool reads the corpus directly and another reads staged slices, "both tools read the
    same corpus" stops being visible in either command line and becomes an invariant nobody is
    checking. Slices cut from corpus A beside a run pointed at corpus B is the comparison-between-
    corpora failure the old run spec guarded against by hand, and it would be silent: every arm
    would finish, every artefact would be present, and the table would be meaningless.

    So the staging step records where the slices came from, and this refuses to proceed on a pair
    that disagree, or on slices that cannot say.
    """
    doc = _read_json(slices / SLICE_PROVENANCE)
    if not doc:
        return [(f"--eval-slices {slices} has no {SLICE_PROVENANCE}, so nothing says which "
                 f"corpus these prompts were cut from and nothing can check they match --track")]
    staged_from = doc.get("track")
    if staged_from is None:
        return [f"{slices / SLICE_PROVENANCE} does not name the corpus it was cut from"]
    if Path(staged_from).resolve() != Path(track).resolve():
        return [(f"the evaluation slices were cut from {staged_from}, and --track is {track}. "
                 f"The two tools would be scored on prompts from different corpora")]
    return []


def write_slice_provenance(slices: Path, track: Path) -> Path:
    """Record which corpus a set of staged slices came from, beside the slices themselves."""
    path = Path(slices) / SLICE_PROVENANCE
    path.write_text(json.dumps({"track": str(Path(track).resolve())}, indent=2) + "\n",
                    encoding="utf-8")
    return path


# ── isolation ─────────────────────────────────────────────────────────────────────────
def docker_available() -> bool:
    return shutil.which("docker") is not None


def isolate_argv(argv, *, image: str, mounts, workdir="/work") -> list[str]:
    """Wrap a command so it runs sealed: no network, read-only inputs, no capabilities.

    The flags are the ones proved by `bench/selftest.py` from inside the box rather than assumed
    from the outside: nine invariants, including that the GPU works and that both mounts refuse a
    write. `--network none` is the one that matters most, because it is what makes a run
    reproducible: nothing can be fetched mid-run that was not staged before it.
    """
    out = ["docker", "run", "--rm", "--network", "none", "--read-only",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "--workdir", workdir]
    for host, guest, mode in mounts:
        out += ["--volume", f"{Path(host).resolve()}:{guest}:{mode}"]
    out += [image, *argv]
    return out


# ── preflight ─────────────────────────────────────────────────────────────────────────
def preflight(*, tools, track: Path, out: Path, model: str, isolate: str, images,
              slices=None, score=False, harmful=None, harmless=None) -> list[str]:
    """Everything checkable before the first GPU second is spent, returned as complaints.

    A long run that dies forty minutes in on something knowable at the start is the most expensive
    kind of failure, and this project has paid it repeatedly: a container that could not write to
    its own output directory, a corpus that was not there, an image missing.
    """
    problems = []
    unknown = [t for t in tools if t not in ADAPTERS]
    if unknown:
        problems.append(
            f"unknown tool(s): {', '.join(unknown)}. Known: {', '.join(sorted(ADAPTERS))}")
    if len(set(tools)) < 2:
        problems.append("a head-to-head needs two different tools; one tool is not a comparison")
    if not Path(track).is_dir():
        problems.append(f"no corpus at {track}")
    out = Path(out)
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".senbon-write-probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        problems.append(f"cannot write to {out}: {e}")
    if not model:
        problems.append("no --model given, so there is nothing to abliterate")
    # A tool that brings its own evaluation prompts is a tool solving a different problem. Both
    # read one staged set, or the comparison is between two searches steered by two rulers.
    needs_slices = [t for t in tools if t in ADAPTERS and t != "senbon"]
    if needs_slices:
        if not slices:
            problems.append(
                f"{', '.join(needs_slices)}: needs --eval-slices, the prompt files both tools "
                f"score on. Without them each tool uses its own and the arms are not comparable")
        else:
            missing = [f for f in SLICE_FILES if not (Path(slices) / f).is_file()]
            if missing:
                problems.append(f"--eval-slices {slices} is missing {', '.join(missing)}")
            problems.extend(slices_match_track(Path(slices), Path(track)))
    # The scoring inputs are checked here rather than after the arms have run, because
    # discovering them missing costs the whole run's GPU time and nothing else.
    if score:
        for flag, path in (("--harmful", harmful), ("--harmless", harmless)):
            if not path:
                problems.append(f"{flag} is needed to score the models; pass it, or --no-score "
                                f"to run the arms and score them later")
            elif not Path(path).exists():
                problems.append(f"{flag} points at {path}, which is not there")
    if isolate == "docker":
        if not docker_available():
            problems.append(
                "--isolate docker was asked for and docker is not on PATH. Install it, or run "
                "with --isolate none and understand that a third-party tool will then run with "
                "your credentials and your network")
        missing = [t for t in tools if t in ADAPTERS and not (images or {}).get(t)]
        if missing:
            problems.append(f"--isolate docker needs an image per tool; none given for: "
                            f"{', '.join(missing)}")
    return problems


# ── the run ───────────────────────────────────────────────────────────────────────────
@dataclass
class ArmResult:
    tool: str
    seed: int
    arm: Path
    ran: bool
    ok: bool
    reason: str
    argv: list = field(default_factory=list)


def default_runner(argv, *, cwd=None, log=print) -> int:
    """Run one arm to completion, streaming its output. Returns the exit code."""
    log(f"  $ {' '.join(str(a) for a in argv)}")
    try:
        proc = subprocess.run(argv, cwd=cwd, check=False)
    except FileNotFoundError as e:
        raise BenchError(f"could not start {argv[0]!r}: {e}") from e
    return proc.returncode


def run_arm(adapter: Adapter, *, seed, model, track, out, trials, extra=(), isolate="none",
            slices=None, image=None, runner=None, log=print, force=False) -> ArmResult:
    """One tool, one seed. Skips itself when an identical arm is already on disk."""
    # Resolved here, not in the signature: a default captured at definition time cannot be
    # substituted, and a test that thought it had replaced the runner started a real subprocess.
    runner = runner or default_runner
    arm = Path(out) / f"{adapter.name}-seed{seed}"
    expected = arm_manifest(adapter, seed, model, trials)

    if not force:
        done, why = arm_is_done(arm, expected, adapter)
        if done:
            log(f"  {adapter.name} seed {seed}: skipping, {why}")
            return ArmResult(adapter.name, seed, arm, ran=False, ok=True, reason=why)

    arm.mkdir(parents=True, exist_ok=True)
    argv = adapter.argv(model=model, track=track, out=arm, seed=seed, trials=trials,
                        slices=slices, extra=list(extra))
    if isolate == "docker":
        argv = isolate_argv(
            argv, image=image,
            mounts=[(model, "/model", "ro"), (track, "/corpus", "ro"), (arm, "/work/out", "rw")])

    log(f"  {adapter.name} seed {seed}: running")
    code = runner(argv, log=log)
    if code != 0:
        return ArmResult(adapter.name, seed, arm, ran=True, ok=False,
                         reason=f"exited {code}", argv=list(argv))

    # EXIT ZERO IS NOT SUCCESS. It says the process ended, not that it produced anything: the
    # 2026-08-05 rehearsal's arms exited 0 having written nothing at all. The artefacts decide.
    missing = [p for p in adapter.produces if not (arm / p).exists()]
    if missing:
        return ArmResult(adapter.name, seed, arm, ran=True, ok=False,
                         reason=f"exited 0 but produced no {', '.join(missing)}",
                         argv=list(argv))
    write_arm_manifest(arm, expected)
    return ArmResult(adapter.name, seed, arm, ran=True, ok=True, reason="produced every artefact",
                     argv=list(argv))


def head_to_head(*, tools, seeds, model, track, out, trials, isolate="none", images=None,
                 extra=(), slices=None, runner=None, log=print, force=False):
    """Every tool, every seed, in order, resuming what is already there.

    Sequential on purpose. Two arms sharing one GPU is how a run dies at 90% with an
    out-of-memory that neither arm caused, and the wall-clock saved would be nothing anyway
    because the card is the bottleneck.
    """
    images = images or {}
    results = []
    for tool in tools:
        adapter = ADAPTERS[tool]
        log(f"{adapter.name}: {len(seeds)} seed(s)")
        for seed in seeds:
            r = run_arm(adapter, seed=seed, model=model, track=track, out=out, trials=trials,
                        extra=extra, isolate=isolate, image=images.get(tool), runner=runner,
                        slices=slices, log=log, force=force)
            results.append(r)
            if not r.ok:
                log(f"  {tool} seed {seed}: FAILED, {r.reason}")
    return results


# ── scoring: one instrument, over every model both tools produced ─────────────────────
def score_argv(*, model: Path, harmful: Path, harmless: Path, out: Path, label: str,
               skip_harmful: int, batch: int) -> list[str]:
    """The compass, invoked the one right way.

    On 2026-08-05 this was invoked three wrong ways at once from a shell loop, and one of them hid
    the other two: it was passed a `--track` the compass does not have, `--skip-harmful` was passed
    with no count so it swallowed the next argument, and the two tools save their models in
    different shapes so one tool's arms were scored against nothing at all. There is one call site
    now and a test asserting each of those three cannot recur.
    """
    return ["python", "-u", "-m", "senbonzakura", "compass",
            "--model", str(model), "--harmful", str(harmful), "--harmless", str(harmless),
            "--out", str(out), "--label", label,
            "--skip-harmful", str(skip_harmful), "--batch", str(batch)]


def arm_model_dir(result: ArmResult, adapter: Adapter) -> Path:
    """Where this tool actually left its model.

    The two disagree and neither is wrong: senbonzakura writes straight into the output directory
    it was given, so the arm directory IS the model, while Heretic's comes out of the selection
    pass in a subdirectory. Assuming one shape scored the other tool's arms against nothing.
    """
    return Path(result.arm) / adapter.model_subdir if adapter.model_subdir else Path(result.arm)


def score_arms(results, *, harmful: Path, harmless: Path, out: Path, skip_harmful=128, batch=16,
               runner=None, log=print, force=False) -> list[dict]:
    """Score every model that exists, and say plainly which ones did not."""
    runner = runner or default_runner
    scored = []
    for r in results:
        adapter = ADAPTERS[r.tool]
        model = arm_model_dir(r, adapter)
        label = f"{r.tool}-seed{r.seed}"
        target = Path(out) / f"scored-{label}.json"
        if not (model / "config.json").is_file():
            log(f"  {label}: NO MODEL at {model}; the arm produced none")
            scored.append({"label": label, "ok": False, "reason": f"no model at {model}"})
            continue
        if target.is_file() and not force:
            log(f"  {label}: already scored")
            scored.append({"label": label, "ok": True, "reason": "already scored",
                           "path": str(target)})
            continue
        code = runner(score_argv(model=model, harmful=harmful, harmless=harmless, out=target,
                                 label=label, skip_harmful=skip_harmful, batch=batch), log=log)
        # Exit zero is not a score. The file is.
        ok = code == 0 and target.is_file()
        scored.append({"label": label, "ok": ok,
                       "reason": "scored" if ok else f"scoring failed (exit {code})",
                       "path": str(target) if ok else None})
        if not ok:
            log(f"  {label}: SCORING FAILED")
    return scored


def summarise(results) -> dict:
    """What happened, in a shape a caller can act on and a reader can check."""
    return {
        "arms": len(results),
        "ran": sum(1 for r in results if r.ran),
        "skipped": sum(1 for r in results if not r.ran),
        "failed": sum(1 for r in results if not r.ok),
        "failures": [{"tool": r.tool, "seed": r.seed, "reason": r.reason}
                     for r in results if not r.ok],
    }


def build_parser():
    ap = argparse.ArgumentParser(
        prog="senbonzakura bench",
        description="Run a head-to-head between abliteration tools on one machine.")
    sub = ap.add_subparsers(dest="operation", required=True)
    h = sub.add_parser("head-to-head", help="run every tool over every seed, then report")
    h.add_argument("--tools", default="senbon,heretic",
                   help=f"comma-separated, from: {', '.join(sorted(ADAPTERS))}")
    h.add_argument("--seeds", default="42,43,44,45,46",
                   help="comma-separated seeds. Fewer than three cannot support a verdict, and "
                        "the report will say so rather than inventing one")
    h.add_argument("--model", required=True, help="the model every arm starts from")
    h.add_argument("--track", required=True, help="the corpus every arm reads. One corpus, or "
                                                 "the comparison is between corpora")
    h.add_argument("--out", required=True, help="where the arms and the report are written")
    h.add_argument("--trials", type=int, default=200,
                   help="search budget per arm, identical for every tool (default: 200)")
    h.add_argument("--isolate", choices=("none", "docker"), default="none",
                   help="'docker' runs each arm sealed: no network, read-only inputs, no "
                        "capabilities. Recommended when running a tool you did not write")
    h.add_argument("--image", action="append", default=[], metavar="TOOL=IMAGE",
                   help="container image for a tool, with --isolate docker. Repeatable")
    h.add_argument("--eval-slices", dest="eval_slices", default="",
                   help="directory of staged prompt files every tool scores on: good.txt, "
                        "bad.txt, keyword_prompts.txt, kl_prompts.txt. Required for any tool "
                        "that would otherwise bring its own evaluation set")
    h.add_argument("--harmful", default="",
                   help="held-out harmful dataset the compass scores every model on. Required "
                        "unless --no-score is given")
    h.add_argument("--harmless", default="",
                   help="held-out harmless dataset for the compass's other arm")
    h.add_argument("--skip-harmful", dest="skip_harmful", type=int, default=128,
                   help="how many harmful rows the search already saw, and the compass must "
                        "therefore skip. A count, never a bare flag (default: 128)")
    h.add_argument("--batch", type=int, default=16,
                   help="scoring batch size, held fixed across every arm so no two arms are "
                        "measured under different conditions")
    h.add_argument("--no-score", dest="score", action="store_false",
                   help="run the arms and stop, leaving scoring and the report for later")
    h.add_argument("--force", action="store_true",
                   help="re-run arms that are already complete instead of skipping them")
    h.add_argument("--arg", action="append", default=[], metavar="ARG",
                   help="extra argument passed through to every arm. Repeatable")

    st = sub.add_parser("stage", help="cut the prompt slices every tool is scored on")
    st.add_argument("--track", required=True, help="the track holding bad_ds / good_ds / bad_eval_ds")
    st.add_argument("--out", required=True, help="directory to write the slices into")
    st.add_argument("--dir-prompts", type=int, default=256)
    st.add_argument("--eval-refusal", type=int, default=64)
    st.add_argument("--eval-refusal-final", type=int, default=128)
    st.add_argument("--eval-kl", type=int, default=64)

    r = sub.add_parser("report", help="read a finished head-to-head and say what it found")
    r.add_argument("run_dir", help="the directory the arms and their scores were written to")
    r.add_argument("--allow-unreadable", action="store_true",
                   help="report over the arms that are readable instead of refusing. An "
                        "incomplete set is not a table, so this has to be asked for")
    return ap


def _parse_images(pairs):
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--image wants TOOL=IMAGE, got {p!r}")
        tool, image = p.split("=", 1)
        out[tool.strip()] = image.strip()
    return out


def _parse_seeds(text):
    try:
        seeds = [int(s) for s in text.split(",") if s.strip()]
    except ValueError as e:
        raise SystemExit(f"--seeds wants whole numbers, got {text!r}") from e
    if not seeds:
        raise SystemExit("--seeds is empty, so there is nothing to run")
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"--seeds repeats a value: {text!r}. A repeated seed is one arm run "
                         f"twice, reported as two independent measurements")
    return seeds


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.operation == "stage":
        from . import benchstage
        raise SystemExit(benchstage.main([
            "--track", a.track, "--out", a.out,
            "--dir-prompts", str(a.dir_prompts), "--eval-refusal", str(a.eval_refusal),
            "--eval-refusal-final", str(a.eval_refusal_final), "--eval-kl", str(a.eval_kl)]))
    if a.operation == "report":
        from . import benchreport
        args = [a.run_dir] + (["--allow-unreadable"] if a.allow_unreadable else [])
        raise SystemExit(benchreport.main(args))
    tools = [t.strip() for t in a.tools.split(",") if t.strip()]
    seeds = _parse_seeds(a.seeds)
    images = _parse_images(a.image)

    slices = Path(a.eval_slices) if a.eval_slices else None
    problems = preflight(tools=tools, track=Path(a.track), out=Path(a.out), model=a.model,
                         isolate=a.isolate, images=images, slices=slices, score=a.score,
                         harmful=a.harmful, harmless=a.harmless)
    if problems:
        print("BENCH REFUSED: nothing was run, because this comparison would not be trustworthy:",
              file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        raise SystemExit(1)

    if a.isolate == "none":
        print("WARNING: arms run unsealed, with your network and your credentials. Pass "
              "--isolate docker to run a third-party tool in a sandbox.", file=sys.stderr)

    results = head_to_head(tools=tools, seeds=seeds, model=a.model, track=Path(a.track),
                           out=Path(a.out), trials=a.trials, isolate=a.isolate, images=images,
                           extra=a.arg, slices=slices, force=a.force)
    summary = summarise(results)
    print(f"BENCH {summary['ran']} ran, {summary['skipped']} skipped, {summary['failed']} failed")
    for f in summary["failures"]:
        print(f"  FAILED {f['tool']} seed {f['seed']}: {f['reason']}")

    if a.score:
        print("scoring every model, with one instrument")
        scored = score_arms([r for r in results if r.ok], harmful=Path(a.harmful),
                            harmless=Path(a.harmless), out=Path(a.out),
                            skip_harmful=a.skip_harmful, batch=a.batch, force=a.force)
        summary["scored"] = scored
        summary["unscored"] = [s["label"] for s in scored if not s["ok"]]

    Path(a.out, "bench-summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                 encoding="utf-8")

    if summary["failed"] or summary.get("unscored"):
        # A partial set is not a table. Say which arms are missing and stop, rather than
        # reporting over whatever happened to survive.
        for label in summary.get("unscored", []):
            print(f"  UNSCORED {label}")
        raise SystemExit(1)

    if a.score:
        from . import benchreport
        benchreport.main([str(a.out)])
    return summary


if __name__ == "__main__":   # pragma: no cover
    main()
