# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Run a head-to-head between abliteration tools, on one machine, in one command.

`senbonzakura head-to-head run --tools senbon,heretic --seeds 5` is the whole thing: it runs
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
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Every arm writes this when it has finished, and the runner checks the file rather than the word.
# A marker in a log is a string, and anything that can print it can claim the work: that is how a
# rehearsal once reported five jobs done having measured nothing.
ARM_MANIFEST = "arm.json"

# Where the sealed box mounts each input. An arm's command is built against THESE rather than
# against this machine's paths, because the two are not the same and the difference is invisible
# until the container starts and cannot find its own model.
GUEST_MODEL = "/model"
GUEST_CORPUS = "/corpus"
GUEST_EVAL = "/corpus-eval"
GUEST_OUT = "/work/out"

# The prompt files every tool is scored on, staged once so no tool brings its own.
SLICE_FILES = ("good.txt", "bad.txt", "keyword_prompts.txt", "final_prompts.txt",
               "kl_prompts.txt")
SLICE_PROVENANCE = "slices.json"


class BenchError(RuntimeError):
    """A failure the operator can act on, phrased for a person rather than a stack trace."""


class ArmTimeoutError(BenchError):
    """One arm hung and was killed. Deliberately distinct from every other BenchError.

    The distinction is about blast radius, not about wording. "Could not start heretic" means
    every remaining arm will fail the same way, so it should stop the sweep; one arm hanging says
    nothing about the arms behind it, and taking them down with it turns a lost arm into a lost
    night. So this one, and only this one, is caught per arm and recorded as a failure.
    """


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
    # An OPTIONAL second pass, run after the tool's own search and before the artefacts are
    # checked, in the same isolation. It exists because one tool cannot finish its own arm:
    # Heretic v1.4.0 ends at an interactive menu it cannot ask in a container, so it never saves,
    # and the equal-budget selection this comparison promises it has to be applied from outside.
    # It lives on the adapter rather than in a runner spec because the adapter is what DECLARES
    # the artefact the pass produces. Split across two files, the declaration outlived the step:
    # decision Q-5 moved the arms into this command and left the pass behind in the spec it
    # replaced, so for five days `produces=("best_of_n.json",)` named a file nothing could write.
    finalise: object = None
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
    return ["python", "-u", "-m", "headtohead.run_heretic",
            "--model", str(model), "--out", str(out),
            "--good", str(s / "good.txt"), "--bad", str(s / "bad.txt"),
            "--keyword-prompts", str(s / "keyword_prompts.txt"),
            "--kl-prompts", str(s / "kl_prompts.txt"),
            "--seed", str(seed), "--trials", str(trials), *extra]


def _heretic_finalise(*, out, slices, **_):
    """The equal-budget selection pass, applied to Heretic from outside its own code.

    senbonzakura does not report the best trial its search found: it re-scores its top six on a
    larger held-out slice and reports the winner of that second look. A comparison that skips the
    equivalent for Heretic measures our selection procedure and calls it our method, which is why
    `headtohead/EQUAL-BUDGET.md` commits us to this pass. It also does the saving, because v1.4.0
    cannot save without a terminal.
    """
    if not slices:
        raise BenchError(
            "heretic's selection pass needs the staged evaluation slices; pass --eval-slices. "
            "They are what make both tools' winners chosen on the same prompts")
    s = Path(slices)
    return ["python", str(_bench_dir_for(out) / "best_of_n_heretic.py"),
            "--out", str(out), "--top-n", "6",
            "--final-prompts", str(s / "final_prompts.txt"),
            "--keyword-prompts", str(s / "keyword_prompts.txt")]


def _bench_dir_for(out) -> Path:
    """Where `headtohead/` is, as the pass will see it.

    Inside the container it is mounted at /work/headtohead; outside it sits beside the package. The
    caller tells us which by the output path it passed, because that is already the guest-or-host
    decision run_arm made.
    """
    if str(out) == GUEST_OUT:
        return Path("/work/headtohead")
    here = Path(__file__).resolve()
    for root in (here.parent.parent.parent, Path.cwd(), Path.home()):
        candidate = root / "headtohead"
        if (candidate / "best_of_n_heretic.py").is_file():
            return candidate
    return Path("headtohead")


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


def _senbon_k_argv(k: int):
    """Our tool with its direction budget PINNED rather than searched.

    The whole claim on the tin is that refusal lives in more than one direction, and the search
    treats the budget as one parameter among nine, so a run that picks its own K measures the
    search rather than the thesis. On 2026-08-12 the five arms of the head-to-head chose K=1 three
    times and K=2 twice, which is a mixture and says nothing either way.

    Pinning it is what turns "does multi-direction help" into an experiment: identical corpus,
    identical budget, identical seeds, one parameter different.
    """
    def build(**kw):
        # BOTH bounds, because --max-directions is a ceiling and the search picks anywhere
        # beneath it. The rehearsal on 2026-08-12 caught this before the real run: the "K=2" arm
        # spent two of its first three trials at K=1 and its frontier was topped by a K=1 config,
        # so it would have shipped a one-direction model under a two-direction label.
        return [*_senbon_argv(**kw), "--min-directions", str(k), "--max-directions", str(k)]
    return build


def _senbon_conv_argv(ablate_conv: bool):
    """Our tool on a hybrid architecture, with the convolution path edited or deliberately not.

    On LFM2 most decoder layers hold a short convolution instead of attention, and its `out_proj`
    writes the residual stream in the same position an attention `o_proj` does. No other
    abliteration tool edits it. On LFM2.5-350M that is 10 of 32 residual writers, 31%.

    The control arm produces a PARTIAL abliteration by construction, which is the point: it is the
    only way to ask whether refusal travels through that path at all. Everything that keeps that
    model from being mistaken for a result lives elsewhere and is deliberate: the run warns per
    layer, `abliteration.json` records `ablate_conv` and the skipped layers, the checkpoint carries
    the same in its config and its safetensors header, and `headtohead report` excludes any arm that
    declares itself partial before a single table is built.
    """
    def build(**kw):
        argv = _senbon_argv(**kw)
        return argv if ablate_conv else [*argv, "--skip-conv-ablation"]
    return build


ADAPTERS: dict[str, Adapter] = {
    "senbon": Adapter(
        name="senbon",
        argv=_senbon_argv,
        produces=("abliteration.json", "config.json"),
        model_subdir="",
        self_report=_senbon_report,
        notes="this tool; writes the model straight into the arm directory",
    ),
    # The controlled arms of the multi-direction experiment. Identical to `senbon` in every
    # respect except the pinned budget, so the pair is a comparison rather than two runs.
    "senbon-k1": Adapter(
        name="senbon-k1",
        argv=_senbon_k_argv(1),
        produces=("abliteration.json", "config.json"),
        model_subdir="",
        self_report=_senbon_report,
        notes="this tool, one direction per layer: the original single-direction method",
    ),
    "senbon-k2": Adapter(
        name="senbon-k2",
        argv=_senbon_k_argv(2),
        produces=("abliteration.json", "config.json"),
        model_subdir="",
        self_report=_senbon_report,
        notes="this tool, two directions per layer: the claim under test",
    ),
    # The controlled arms of the hybrid experiment (task 40). Identical to `senbon` and to each
    # other in every respect except whether the convolution output projections are edited.
    "senbon-conv": Adapter(
        name="senbon-conv",
        argv=_senbon_conv_argv(True),
        produces=("abliteration.json", "config.json"),
        model_subdir="",
        self_report=_senbon_report,
        notes="this tool, whole: attention, MLP and the convolution output projections",
    ),
    "senbon-noconv": Adapter(
        name="senbon-noconv",
        argv=_senbon_conv_argv(False),
        produces=("abliteration.json", "config.json"),
        model_subdir="",
        self_report=_senbon_report,
        notes="CONTROL ARM, a partial abliteration: the convolution path is left untouched",
    ),
    "heretic": Adapter(
        name="heretic",
        argv=_heretic_argv,
        produces=("best_of_n.json",),
        model_subdir="model",
        self_report=_heretic_report,
        finalise=_heretic_finalise,
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

    Used when `headtohead/run-isolated.sh` is not available. It is the minimum sealed box and it is
    deliberately not the one this project runs its own arms in: see `isolation_wrapper`.
    """
    out = ["docker", "run", "--rm", "--network", "none", "--read-only",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "--workdir", workdir]
    for host, guest, mode in mounts:
        out += ["--volume", f"{Path(host).resolve()}:{guest}:{mode}"]
    out += [image, *argv]
    return out


def find_run_isolated() -> Path | None:
    """`headtohead/run-isolated.sh`, if this is a checkout rather than an installed wheel."""
    override = os.environ.get("SENBON_RUN_ISOLATED")
    if override:
        p = Path(override)
        return p if p.is_file() else None
    here = Path(__file__).resolve()
    # A checkout has it beside the package; a machine the code was shipped to has it wherever the
    # shipping put it, which on this project's card is ~/headtohead rather than beside the source.
    roots = (here.parent.parent.parent, Path.cwd(), Path.home())
    for root in roots:
        candidate = root / "headtohead" / "run-isolated.sh"
        if candidate.is_file():
            return candidate
    return None


def isolation_wrapper(script: Path, argv, *, tool, image, model, track, slices, out) -> list[str]:
    """Run an arm through `headtohead/run-isolated.sh`, which owns the isolation.

    THE FLAGS ARE NOT REIMPLEMENTED HERE, and that is the point.

    A sealed box for this benchmark needs more than "no network, read-only, no capabilities". On
    WSL2, a GPU inside a container needs `/dev/dxg` AND `/usr/lib/wsl/lib` AND
    `/usr/lib/wsl/drivers`; miss the driver store and libcuda loads, reports that it cannot
    initialise NVML, and returns zero devices, which reads as "no GPU here" rather than "one bind
    mount short". It also needs the corpus, the staged eval slices, our package source so a pass
    running inside can import the shared ruler, and the headtohead directory so the tool's own entry
    point is importable.

    That is seven mounts and a device, and `headtohead/selftest.py` verifies nine invariants about them
    from INSIDE the box. Rebuilding that list in Python would mean maintaining two copies of a
    thing this project has already been bitten by having three copies of, and the copy that drifts
    is the one nobody re-verifies.
    """
    return [str(script), "--tool", tool, "--image", image,
            "--model", str(model), "--corpus", str(track), "--out", str(out),
            *(["--eval", str(slices)] if slices else []),
            "--senbon-src", str(Path(__file__).resolve().parent.parent),
            "--", *argv]


# ── preflight # ── preflight ─────────────────────────────────────────────────────────────────────────
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


#: Hours an arm may run before it is killed. An arm is a whole abliteration search, so this is
#: generous by design: the failure that cost a completed head-to-head on 2026-08-06 was a job
#: killed at a ceiling with its work already done, and that is worse than waiting. What this
#: bounds is the OTHER failure, a hung process holding a sweep open indefinitely.
ARM_TIMEOUT_S = 12 * 3600


def check_argv(argv) -> None:
    """Refuse to launch a command whose flag has been handed the word "None".

    THE LAST LINE OF DEFENCE FOR A WHOLE CLASS OF DEFECT. Roughly thirty places in this codebase
    build an argv by interpolating `str(value)` next to a flag, and `str()` renders anything: a
    `None` that means "not set" becomes the four characters N-o-n-e, the child's argparse refuses
    them, and the run dies at parsing after the expensive part is already done. It cost a
    108-minute ten-arm comparison its entire scoring pass.

    Fixing the one composer that did it fixes one composer. This is at the choke point every
    composer goes through, so a new one written next month is covered without anybody
    remembering. Only a value directly after a `--flag` is checked, because a bare "None" can be
    a legitimate label or a filename, and refusing those would be a worse rule than no rule.
    """
    tokens = [str(a) for a in argv]
    for i, word in enumerate(tokens[1:], 1):
        if word == "None" and tokens[i - 1].startswith("--"):
            raise BenchError(
                f"the command being launched passes {tokens[i - 1]} the word 'None', which means "
                f"a value that was never resolved has been rendered as text. The child would "
                f"refuse it at argument parsing, after the expensive part of this run. Resolve "
                f"the value or omit the flag.\n  $ {' '.join(tokens)}")


def default_runner(argv, *, cwd=None, log=print, timeout=ARM_TIMEOUT_S) -> int:
    """Run one arm to completion, streaming its output. Returns the exit code.

    The timeout is not optional and has no "wait forever" setting, because a sweep is a sequence
    and one hung arm stops every arm behind it. holst-orchestrated runs carry their own
    `timeout_secs` and `stall_secs`, so this covers the case those do not: `senbonzakura head-to-head`
    invoked directly, where nothing else is watching. A `llama-cli` smoke hung for two hours on
    2026-08-16 with no bound on it at all, which is this failure one layer down.

    On expiry the child is killed and the arm is reported as failed rather than as never-run, so
    the sweep continues and the artefact records which arm died and why.
    """
    check_argv(argv)
    log(f"  $ {' '.join(str(a) for a in argv)}")
    try:
        # The child's stderr is merged into ITS STDOUT, so both follow the parent's stdout to
        # wherever the run is being logged. Left separate, a harness redirecting only stdout to a
        # file sends every diagnostic somewhere else: a ten-arm run failed on a one-line argparse
        # error that was printed, went to a different stream from the log, and had to be
        # rediscovered by re-running the command by hand. The failure path is where the output
        # matters most and it was the one path where it went missing.
        proc = subprocess.run(argv, cwd=cwd, check=False, timeout=timeout,
                              stderr=subprocess.STDOUT)
    except FileNotFoundError as e:
        raise BenchError(f"could not start {argv[0]!r}: {e}") from e
    except subprocess.TimeoutExpired as e:
        # subprocess.run has already killed the child by the time this is raised.
        log(f"  TIMED OUT after {timeout / 3600:.1f}h and was killed: {argv[0]}")
        raise ArmTimeoutError(
            f"ran for {timeout / 3600:.1f} hours without finishing and was killed. An arm that "
            f"exceeds this is hung rather than slow; the usual causes are a model that will not "
            f"generate, a prompt for input on a non-interactive stream, or a deadlocked "
            f"dataloader. Re-run that arm alone to see where it stops.") from e
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

    # THE ARM IS GIVEN THE PATHS IT WILL SEE, NOT THE ONES WE SEE.
    #
    # A container mounts the model at /model and the corpus at /corpus, so an arm handed
    # `--model /home/<user>/headtohead-models/Qwen3-1.7B` looks for a directory that does not exist
    # inside the box and dies on its first line. The rehearsal on 2026-08-10 caught exactly that:
    # every arm failed instantly with "No module named senbonzakura", because the command was
    # built against this machine's layout and run somewhere with a different one.
    if isolate == "docker":
        paths = dict(model=GUEST_MODEL, track=GUEST_CORPUS, out=GUEST_OUT,
                     slices=(GUEST_EVAL if slices else None))
    else:
        paths = dict(model=model, track=track, out=arm, slices=slices)

    argv = adapter.argv(seed=seed, trials=trials, extra=list(extra), **paths)
    if isolate == "docker":
        script = find_run_isolated()
        if script is not None:
            argv = isolation_wrapper(script, argv, tool=adapter.name, image=image, model=model,
                                     track=track, slices=slices, out=arm)
        else:
            argv = isolate_argv(
                argv, image=image,
                mounts=[(model, "/model", "ro"), (track, "/corpus", "ro"),
                        (arm, "/work/out", "rw")])

    log(f"  {adapter.name} seed {seed}: running")
    # WHEN THIS ARM STARTED, so the artefact check can tell what this arm produced from what was
    # simply lying in the directory. See the staleness check below.
    started = time.time() - 1          # a second of slack for filesystem timestamp granularity
    try:
        code = runner(argv, log=log)
    except ArmTimeoutError as e:
        # This arm only. Every other BenchError still propagates, because the rest are conditions
        # that will meet the next arm identically.
        return ArmResult(adapter.name, seed, arm, ran=True, ok=False,
                         reason=f"timed out: {e}", argv=list(argv))
    if code != 0:
        return ArmResult(adapter.name, seed, arm, ran=True, ok=False,
                         reason=f"exited {code}", argv=list(argv))

    # THE SECOND PASS, when the tool cannot finish its own arm.
    #
    # Heretic ends at an interactive menu, so its search leaves a complete study and no model. The
    # pass below reads that study, applies the same best-of-N selection senbonzakura applies to
    # itself, and saves the winner. Without it the arm runs for the full budget and produces
    # nothing to score, which is precisely what 2026-08-11 spent four hours doing.
    if adapter.finalise is not None:
        final_argv = adapter.finalise(**paths)
        if isolate == "docker":
            script = find_run_isolated()
            if script is not None:
                final_argv = isolation_wrapper(
                    script, final_argv, tool=f"{adapter.name}-best-of-n", image=image,
                    model=model, track=track, slices=slices, out=arm)
            else:
                final_argv = isolate_argv(
                    final_argv, image=image,
                    mounts=[(model, "/model", "ro"), (track, "/corpus", "ro"),
                            (arm, "/work/out", "rw")])
        log(f"  {adapter.name} seed {seed}: selection pass")
        code = runner(final_argv, log=log)
        if code != 0:
            return ArmResult(adapter.name, seed, arm, ran=True, ok=False,
                             reason=f"the selection pass exited {code}", argv=list(final_argv))

    # EXIT ZERO IS NOT SUCCESS. It says the process ended, not that it produced anything: the
    # 2026-08-05 rehearsal's arms exited 0 having written nothing at all. The artefacts decide.
    #
    # AND AN ARTEFACT THIS ARM DID NOT WRITE IS NOT THIS ARM'S ARTEFACT. Existence alone was the
    # check until 2026-08-11, and it let a rehearsal pass on a model five days old: the tool ran,
    # saved nothing, and the previous run's output was still in the directory, so the check found
    # what it was looking for and the arm was written up as a success with a fresh manifest
    # vouching for it. The rehearsal exists to prove the pipeline works and it was reading files
    # from the pipeline it was meant to be testing.
    missing, stale = [], []
    for name in adapter.produces:
        artefact = arm / name
        if not artefact.exists():
            missing.append(name)
        elif artefact.stat().st_mtime < started:
            stale.append(name)
    if missing or stale:
        why = []
        if missing:
            why.append(f"produced no {', '.join(missing)}")
        if stale:
            why.append(f"left {', '.join(stale)} untouched from an earlier run, so this arm "
                       f"produced nothing and would have been scored on stale output")
        return ArmResult(adapter.name, seed, arm, ran=True, ok=False,
                         reason="exited 0 but " + " and ".join(why), argv=list(argv))
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
               skip_harmful: int, batch: int, track: Path | None = None) -> list[str]:
    """The compass, invoked the one right way.

    On 2026-08-05 this was invoked three wrong ways at once from a shell loop, and one of them hid
    the other two: it was passed a `--track` the compass did not have, `--skip-harmful` was passed
    with no count so it swallowed the next argument, and the two tools save their models in
    different shapes so one tool's arms were scored against nothing at all. There is one call site
    now and a test asserting each of those three cannot recur.

    The first of those three has since changed shape rather than gone away. As of 2026-08-17 the
    compass DOES take `--track`, and passing it is now the correct thing to do: the manifest is
    where the partition boundaries actually fell, and `--skip-harmful` alone is a restatement of
    them that was measurably wrong (128 against a 132-row selection partition). So the rule is no
    longer "never pass a track", it is "pass the real one, and let it beat the default". A track
    that is not a track is still refused, by the compass, on the manifest it cannot read.
    """
    # `sys.executable`, NOT "python". The scorer runs on the HOST, unlike the arms, which run
    # inside a container where `python` exists. The card this benchmark runs on has `python3` and
    # no `python` at all, so a literal "python" dies with "could not start 'python'" the moment
    # scoring is actually reached. It went unnoticed because scoring had always found a previous
    # run's results and reported "already scored"; the first rehearsal with a genuinely empty
    # output directory hit it immediately (2026-08-11).
    #
    # It is also the correct interpreter on its own merits: the compass must run with the same
    # senbonzakura the harness was started from, or `-m senbonzakura` resolves to a different
    # install than the one being tested.
    argv = [sys.executable, "-u", "-m", "senbonzakura", "compass",
            "--model", str(model), "--harmful", str(harmful), "--harmless", str(harmless),
            "--out", str(out), "--label", label, "--batch", str(batch)]
    if track is not None:
        argv += ["--track", str(track)]
    # Only when explicitly set. Left off, the compass reads the boundary from the track, which is
    # the point; passing both a track and a contradicting count is refused there rather than here,
    # so the refusal names the manifest that disagrees.
    if skip_harmful is not None:
        argv += ["--skip-harmful", str(skip_harmful)]
    return argv


def arm_model_dir(result: ArmResult, adapter: Adapter) -> Path:
    """Where this tool actually left its model.

    The two disagree and neither is wrong: senbonzakura writes straight into the output directory
    it was given, so the arm directory IS the model, while Heretic's comes out of the selection
    pass in a subdirectory. Assuming one shape scored the other tool's arms against nothing.
    """
    return Path(result.arm) / adapter.model_subdir if adapter.model_subdir else Path(result.arm)


def score_arms(results, *, harmful: Path, harmless: Path, out: Path, skip_harmful=None, batch=16,
               runner=None, log=print, force=False, track=None) -> list[dict]:
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
                                 label=label, skip_harmful=skip_harmful, batch=batch,
                                 track=track), log=log)
        # Exit zero is not a score. The file is.
        ok = code == 0 and target.is_file()
        scored.append({"label": label, "ok": ok,
                       "reason": "scored" if ok else f"scoring failed (exit {code})",
                       "path": str(target) if ok else None})
        if not ok:
            log(f"  {label}: SCORING FAILED")
    return scored


def drift_argv(*, model: Path, base: Path, prompts: Path, out: Path, label: str,
               batch: int, cache: Path) -> list[str]:
    """The one coherence ruler, invoked the one right way.

    `sys.executable` for the same reason the compass uses it: this runs on the host that dispatches
    the arms, not inside a tool's container, and the card has no `python`.
    """
    return [sys.executable, "-u", "-m", "senbonzakura", "drift",
            "--model", str(model), "--base", str(base), "--prompts", str(prompts),
            "--out", str(out), "--label", label, "--batch", str(batch),
            "--base-cache", str(cache)]


DRIFT_SKIP_HARMLESS = 320
DRIFT_EVAL_N = 200


def drift_prompt_slice(harmless: Path, out: Path, skip=DRIFT_SKIP_HARMLESS, n=DRIFT_EVAL_N,
                       log=print) -> Path:
    """The harmless prompts drift is measured on: the ones NEITHER tool has seen.

    The obvious slice to reuse was `kl_prompts.txt`, and it is the wrong one. That file is handed
    to Heretic during its search as the set its own KL is computed on, so measuring drift there
    asks each tool how it did on prompts one of them tuned against. It biased toward Heretic and
    Heretic still came out worse, but a confound that happens to point the other way is still a
    confound.

    So this takes harmless rows after the same skip the compass uses, which is the measure
    partition: held out from direction fitting by the track's own boundaries, and never shown to
    either tool. All three axes then read the same exam.
    """
    from .headtohead_stage import load_texts

    target = Path(out) / "drift-prompts.txt"
    prompts = load_texts(str(harmless), skip + n)[skip:skip + n]
    if len(prompts) < n:
        log(f"  NOTE only {len(prompts)} harmless rows are available after a skip of {skip}; "
            f"drift is measured on those")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    log(f"  drift is measured on {len(prompts)} harmless prompts neither tool has seen")
    return target


def drift_arms(results, *, base: Path, prompts: Path, out: Path, batch=16,
               runner=None, log=print, force=False) -> list[dict]:
    """Measure every model's drift from the base with ONE instrument, after the fact.

    THE AXIS THE BENCHMARK WAS MISSING. Each tool reports a KL of its own, computed on its own
    slice during its own search, and on 2026-08-12 those two numbers came out a hundredfold apart
    with no honest sentence available across them. `score_arms` solved exactly this shape for harm
    recognition; this is the same answer for coherence, and the two together are what make the
    comparison a comparison rather than two self-reports side by side.

    Every model is measured on the SAME prompts against the SAME base at the SAME batch size,
    because all three change the number.
    """
    runner = runner or default_runner
    measured = []
    cache = Path(out) / "drift-base-logprobs.pt"
    for r in results:
        adapter = ADAPTERS[r.tool]
        model = arm_model_dir(r, adapter)
        label = f"{r.tool}-seed{r.seed}"
        target = Path(out) / f"drift-{label}.json"
        if not (model / "config.json").is_file():
            log(f"  {label}: NO MODEL at {model}; the arm produced none")
            measured.append({"label": label, "ok": False, "reason": f"no model at {model}"})
            continue
        if target.is_file() and not force:
            log(f"  {label}: drift already measured")
            measured.append({"label": label, "ok": True, "reason": "already measured",
                             "path": str(target)})
            continue
        code = runner(drift_argv(model=model, base=base, prompts=prompts, out=target,
                                 label=label, batch=batch, cache=cache), log=log)
        # Exit zero is not a measurement. The file is.
        ok = code == 0 and target.is_file()
        measured.append({"label": label, "ok": ok,
                         "reason": "measured" if ok else f"drift failed (exit {code})",
                         "path": str(target) if ok else None})
        if not ok:
            log(f"  {label}: DRIFT MEASUREMENT FAILED")
    return measured


# THE SAME ROWS THE COMPASS READS. The compass evaluates 200 harmful prompts after the same skip,
# and matching it is what makes "one exam" true rather than a turn of phrase. Left at `--n 0` this
# scores every remaining row, which on this project's corpus is 4,504 of them: forty-five minutes a
# model, seven and a half hours for a table, and measured on different rows from the other two axes.
REFUSAL_EVAL_N = 200


def refusal_argv(*, model: Path, harmful: Path, out: Path, label: str,
                 skip: int, batch: int, n: int = REFUSAL_EVAL_N) -> list[str]:
    """The refusal ruler, invoked the one right way, on the host for the same reason as the rest.

    THE COUNTS ARE CHECKED HERE, and the reason is a 108-minute run that produced nothing.
    `--skip-harmful` defaults to None, meaning "read the boundary from the track". The compass
    path handles that by omitting the flag; this one interpolated it, so the child received the
    literal string `None` and argparse refused it in under a second, ten times, after every arm
    had already been trained. `str()` will render anything, which is exactly the problem: the
    same value was correct in one composer and nonsense in the other, and nothing compared them.

    Omitting the flag is NOT the fallback here, because `score --skip` defaults to 0 and would
    silently count refusals on the rows the search selected on. That is the contamination this
    project exists to detect. So an unresolved boundary is a refusal, not a default.
    """
    for name, value in (("skip", skip), ("n", n), ("batch", batch)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise BenchError(
                f"refusal scoring was handed {name}={value!r}, which is not a count. Interpolating "
                f"it would pass the child the text {str(value)!r} and every arm would fail at "
                f"argument parsing after being trained. If this is the held-out boundary, resolve "
                f"it from the track's manifest before scoring.")
    return [sys.executable, "-u", "-m", "senbonzakura", "score",
            "--model", str(model), "--eval", str(harmful), "--out", str(out),
            "--label", label, "--skip", str(skip), "--n", str(n), "--batch", str(batch)]


def refusal_arms(results, *, harmful: Path, out: Path, skip=128, batch=16,
                 n=REFUSAL_EVAL_N,
                 runner=None, log=print, force=False) -> list[dict]:
    """Count every model's refusals with ONE ruler on ONE slice.

    THE LAST AXIS TO BECOME COMPARABLE, and the one the tool exists to move. The compass made
    harm recognition comparable and `drift_arms` made coherence comparable, and refusal was still
    two self-reports: both computed with senbonzakura's rulers, and both on different prompts, one
    on our search evaluation and one on the shared re-score slice. Same ruler, different exam, so
    still not a column.

    Without this the interesting sentence cannot be written at all. "Removed at least as many
    refusals AND drifted less" needs both halves measured the same way, and until now only the
    drift half was.
    """
    runner = runner or default_runner
    measured = []
    for r in results:
        adapter = ADAPTERS[r.tool]
        model = arm_model_dir(r, adapter)
        label = f"{r.tool}-seed{r.seed}"
        target = Path(out) / f"refusal-{label}.json"
        if not (model / "config.json").is_file():
            log(f"  {label}: NO MODEL at {model}; the arm produced none")
            measured.append({"label": label, "ok": False, "reason": f"no model at {model}"})
            continue
        if target.is_file() and not force:
            log(f"  {label}: refusals already counted")
            measured.append({"label": label, "ok": True, "reason": "already counted",
                             "path": str(target)})
            continue
        code = runner(refusal_argv(model=model, harmful=harmful, out=target, label=label,
                                   skip=skip, batch=batch, n=n), log=log)
        # Exit zero is not a count. The file is.
        ok = code == 0 and target.is_file()
        measured.append({"label": label, "ok": ok,
                         "reason": "counted" if ok else f"refusal scoring failed (exit {code})",
                         "path": str(target) if ok else None})
        if not ok:
            log(f"  {label}: REFUSAL SCORING FAILED")
    return measured


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
        prog="senbonzakura head-to-head",
        description="Run a head-to-head between abliteration tools on one machine.")
    sub = ap.add_subparsers(dest="operation", required=True)
    h = sub.add_parser("run", help="run every tool over every seed, then report")
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
    h.add_argument("--skip-harmful", dest="skip_harmful", type=int, default=None,
                   help="how many harmful rows the search already saw, and the compass must "
                        "therefore skip. A count, never a bare flag. Left unset, the compass "
                        "reads the boundary out of the track's own manifest, which is the "
                        "recorded fact rather than a restatement of it")
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
        from . import headtohead_stage
        raise SystemExit(headtohead_stage.main([
            "--track", a.track, "--out", a.out,
            "--dir-prompts", str(a.dir_prompts), "--eval-refusal", str(a.eval_refusal),
            "--eval-refusal-final", str(a.eval_refusal_final), "--eval-kl", str(a.eval_kl)]))
    if a.operation == "report":
        from . import headtohead_report
        args = [a.run_dir] + (["--allow-unreadable"] if a.allow_unreadable else [])
        raise SystemExit(headtohead_report.main(args))
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
                            skip_harmful=a.skip_harmful, batch=a.batch, force=a.force,
                            track=a.track)
        summary["scored"] = scored
        summary["unscored"] = [s["label"] for s in scored if not s["ok"]]

        # THE THIRD AXIS. Harm recognition alone leaves coherence as two self-reports that cannot
        # be read across, which is what the 2026-08-12 table had to admit. One base, one prompt
        # slice, one batch size, every model.
        print("measuring drift from the base, with one instrument")
        drifted = drift_arms([r for r in results if r.ok], base=Path(a.model),
                             prompts=drift_prompt_slice(Path(a.harmless), Path(a.out)),
                             out=Path(a.out), batch=a.batch, force=a.force)
        summary["drift"] = drifted
        summary["undrifted"] = [d["label"] for d in drifted if not d["ok"]]

        # THE AXIS THE TOOL EXISTS TO MOVE, finally on one ruler and one slice. The same
        # held-out rows the compass reads, so all three axes describe the same exam.
        print("counting refusals, with one ruler on one slice")
        # Resolved HERE rather than passed through, because `score` has no `--track` and cannot
        # read the boundary for itself the way the compass does. Same resolver the compass uses,
        # so the two axes cannot end up reading different rows: a second copy of this arithmetic
        # is how three copies of the prompt renderer drifted and put the compass's read-out on
        # the wrong token.
        from .margin import resolve_skips
        skip_harmful, _ = resolve_skips(a.track, a.skip_harmful, None, log=print)
        refused = refusal_arms([r for r in results if r.ok], harmful=Path(a.harmful),
                               out=Path(a.out), skip=skip_harmful, batch=a.batch,
                               force=a.force)
        summary["refusal"] = refused
        summary["uncounted"] = [x["label"] for x in refused if not x["ok"]]

    Path(a.out, "headtohead-summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                 encoding="utf-8")

    if (summary["failed"] or summary.get("unscored") or summary.get("undrifted")
            or summary.get("uncounted")):
        # A partial set is not a table. Say which arms are missing and stop, rather than
        # reporting over whatever happened to survive.
        for label in summary.get("unscored", []):
            print(f"  UNSCORED {label}")
        for label in summary.get("undrifted", []):
            print(f"  NO DRIFT MEASUREMENT {label}")
        for label in summary.get("uncounted", []):
            print(f"  NO REFUSAL COUNT {label}")
        raise SystemExit(1)

    if a.score:
        from . import headtohead_report
        headtohead_report.main([str(a.out)])
    return summary


if __name__ == "__main__":   # pragma: no cover
    # Through `exit_status` so `python -m senbonzakura.<module>` reports what the console
    # script reports. A bare `main()` discards the return, which is how `doctor` printed
    # nine failed checks and exited 0; `sys.exit(main())` alone breaks the other way for
    # the commands that return their result rather than a status.
    import sys

    from .entry import exit_status
    sys.exit(exit_status(main()))
