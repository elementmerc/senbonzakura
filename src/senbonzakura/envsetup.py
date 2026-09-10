# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`senbonzakura setup`: put the right build of torch on THIS machine.

WHY A COMMAND RATHER THAN PACKAGING

pip cannot do this. Environment markers (PEP 508) expose the interpreter, the operating system and
the machine architecture, and there is no marker for "has an NVIDIA GPU". A wheel executes no code
at install time, so it cannot look. And PyPI forbids depending on an external index, which is where
every CUDA-variant and CPU-variant build of torch actually lives.

So what `pip install senbonzakura` gets you is decided by platform alone, and platform is a poor
proxy for hardware:

    Linux x86_64    torch's CUDA build, plus 15 nvidia-* packages, whether or not a card exists
    Linux aarch64   the same CUDA packages, on hardware where they are almost certainly useless
    macOS arm64     a 127 MB build that uses Metal, which is correct
    Windows amd64   a 124 MB CPU-ONLY build, on a machine that may well have a 3060 in it

That last line is the one that matters. A Windows laptop with a GPU installs a torch that cannot
see it, and nothing says so: the card simply sits idle while the search runs on CPU and takes a day
instead of an hour.

WHAT THIS COMMAND DOES ABOUT IT

Looks at the machine, says what it found, and names the exact command that fixes it. With
`--apply` it runs that command. Without it, it changes nothing: a tool that rewrites the
environment it was pointed at, because it believed it knew better, is worse than a wrong answer.

THE DETECTION TRAP THIS AVOIDS

`torch.cuda.is_available()` is the obvious way to look for a GPU and it is the wrong question. It
answers "can THIS BUILD of torch reach a GPU", which on a CPU-only wheel is False on a machine with
a card in it. That is precisely the situation this command exists to find, so asking torch would
make it blind to the only fault it was written for. The driver is asked instead, through
`nvidia-smi`, which is a fact about the machine rather than about what we installed on it.
"""
from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys

#: PyTorch's own channels, newest first. A build for a given CUDA version needs a driver that
#: supports at least that version, so the pick is the newest channel the driver can carry.
#: Read from download.pytorch.org rather than invented; `cu132` is the newest at time of writing.
CUDA_CHANNELS: tuple[tuple[tuple[int, int], str], ...] = (
    ((13, 2), "cu132"),
    ((13, 0), "cu130"),
    ((12, 9), "cu129"),
    ((12, 8), "cu128"),
    ((12, 6), "cu126"),
    ((12, 4), "cu124"),
    ((12, 1), "cu121"),
    ((11, 8), "cu118"),
)

INDEX = "https://download.pytorch.org/whl"

#: The oldest GPU each channel's wheels actually carry kernels for, as a compute capability.
#: A driver's CUDA ceiling is necessary and NOT sufficient: a GTX 1080 (sm_61) on a current driver
#: reports CUDA 13.x, so the driver says yes while the wheel has no kernel for the card. The
#: failure is `no kernel image is available for execution on the device`, minutes into a run, which
#: is exactly the late failure this command exists to move forward. Approximate by design, and
#: deliberately generous: this warns, it does not refuse.
CHANNEL_ARCH_FLOOR: dict[str, tuple[int, int]] = {
    "cu132": (7, 5), "cu130": (7, 5),
    "cu129": (5, 0), "cu128": (5, 0), "cu126": (5, 0), "cu124": (5, 0), "cu121": (5, 0),
    "cu118": (3, 7),
}

#: A channel name we are willing to build an index URL from without having seen it in the table.
#: ROCm and nightly channels are real and legitimately outside `CUDA_CHANNELS`, so they warn; a
#: `cu`-shaped name that is not one we know is almost always a typo and is refused, because the
#: whole argument for `tools/check_cuda_channels.py` is that a URL which 404s is worse than no
#: advice, and `--cuda` walked straight past it.
_CHANNEL_SHAPE = re.compile(r"^(?:cpu|cu\d{3,4}|rocm[0-9.]+|nightly(?:/[a-z0-9.]+)?)$")

#: A pip install of torch is hundreds of megabytes and can stall on a half-open connection. Every
#: other subprocess in this module is bounded; baseline 2.1 says this one must be too.
PIP_TIMEOUT = 3600

#: When the table above was last checked against the index, and how long that is good for. The
#: same shape as `vendor/pins.json`: a hardcoded list of somebody else's versions rots, and a list
#: that rots silently is worse than one that says how old it is. `tools/check_cuda_channels.py`
#: does the network half at release time; the arithmetic here is pure so it can be tested.
CHANNELS_CHECKED = "2026-09-10"
CHANNELS_STALE_AFTER_DAYS = 90


def channels_age_days(today, checked=None):
    """Days since the channel table was verified against the index."""
    from datetime import date
    checked = checked or CHANNELS_CHECKED
    y, m, d = (int(x) for x in checked.split("-"))
    return (today - date(y, m, d)).days


def channels_are_stale(today, checked=None, limit=None):
    return channels_age_days(today, checked) > (limit or CHANNELS_STALE_AFTER_DAYS)


def channel_problems(published, table=None):
    """What is wrong with our table given what the index actually publishes.

    Two different faults, reported separately because they need different fixes. A channel we
    name that is NOT published is the dangerous one: the command prints an index URL that 404s,
    and the user finds out by running it. A channel published that we do NOT name is only a
    missed opportunity, and it means a new card gets an older build than it could have.
    """
    table = table or CUDA_CHANNELS
    ours = [tag for _, tag in table]
    gone = [tag for tag in ours if tag not in published]
    newer = sorted(
        (c for c in published
         if c.startswith("cu") and c[2:].isdigit() and int(c[2:]) > max(int(t[2:]) for t in ours)),
        key=lambda c: int(c[2:]))
    return {"gone": gone, "newer": newer}

def check_channel(channel, table=None):
    """(accepted, complaint) for a channel name given to `--cuda`.

    `--cuda` used to be interpolated into the index URL untouched, so `--cuda cu13O` (capital O)
    printed a confident install command against a channel that does not exist. That is the exact
    outcome `tools/check_cuda_channels.py` was written to prevent, reached through the one path
    that never consulted the table.
    """
    known = {tag for _, tag in (table or CUDA_CHANNELS)} | {"cpu"}
    if channel in known:
        return True, None
    if not _CHANNEL_SHAPE.match(channel or ""):
        return False, (f"`{channel}` is not a channel name PyTorch publishes. Known CUDA channels "
                       f"are {', '.join(sorted(known - {'cpu'}))}, plus `cpu`. A ROCm or nightly "
                       f"channel is also accepted; see https://pytorch.org/get-started/locally/.")
    return True, (f"`{channel}` is not in this tool's table (last checked {CHANNELS_CHECKED}), so "
                  f"the index URL below is unverified. If it 404s, that is why.")


#: What a torch version string tells us about the build it came from. `2.14.0+cpu` and
#: `2.14.0+cu130` are explicit; a bare `2.14.0` is whatever the platform's default wheel is, which
#: is the case that needs the platform table above to interpret it.
_LOCAL = re.compile(r"\+(?P<tag>[a-z][a-z0-9.]*)$")


def nvidia_gpus():
    """GPU names from the DRIVER, or None when the question could not be asked.

    None is not "no GPU". A machine with no `nvidia-smi` might have no NVIDIA card, or might have
    one whose driver is not installed, and those need different advice. Saying "none found" for
    both is the kind of confident wrong answer this project keeps having to withdraw.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip().startswith("GPU ")]


def driver_cuda_version():
    """The highest CUDA version this driver supports, as (major, minor), or None.

    This is NOT the CUDA toolkit version installed on the machine, and it is not what torch was
    built against. It is the ceiling: a driver reporting 13.0 runs any build up to 13.0.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    # A driver in a partial-failure state can print a header and still exit non-zero. `nvidia_gpus`
    # checks this and this did not, so a number from a run that failed would have been believed.
    if out.returncode != 0:
        return None
    # Windows nvidia-smi prints "CUDA UMD Version: 13.3"; Linux prints "CUDA Version: 13.0".
    # Measured on the ROG 2026-09-10, where the first version of this regex read neither and
    # the unread value then became a claim that the driver was too old.
    m = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", out.stdout or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def compute_capability():
    """The OLDEST card's compute capability as (major, minor), or None when it cannot be asked.

    The oldest, because a build has to carry kernels for every card it will be asked to run on,
    and the weakest one decides.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    caps = []
    for line in (out.stdout or "").splitlines():
        m = re.match(r"\s*(\d+)\.(\d+)\s*$", line)
        if m:
            caps.append((int(m.group(1)), int(m.group(2))))
    return min(caps) if caps else None


def card_is_too_old_for(channel, compute):
    """A complaint when the card cannot run the channel's kernels, else None."""
    floor = CHANNEL_ARCH_FLOOR.get(channel)
    if floor is None or compute is None or compute >= floor:
        return None
    return (f"the oldest card here is compute capability {compute[0]}.{compute[1]} and `{channel}` "
            f"wheels carry kernels from {floor[0]}.{floor[1]} up, so that build would load and "
            f"then fail at the first kernel launch with `no kernel image is available`. An older "
            f"channel is the fix; pass --cuda to choose one.")


def pick_cuda_channel(driver, compute=None):
    """The newest channel this machine can actually run, or None when it can run none of them.

    Two constraints, not one. The driver sets a ceiling on the CUDA version, and the card sets a
    floor on the kernels the wheel has to contain. Asking only the driver recommends a cu13x build
    to a Pascal card, which installs cleanly and dies at the first kernel launch.
    """
    if driver is None:
        return None
    for need, tag in CUDA_CHANNELS:
        if need <= driver and card_is_too_old_for(tag, compute) is None:
            return tag
    return None


def installed_torch():
    """(version, variant) of the installed torch, or (None, None).

    Read from the distribution metadata rather than by importing torch: importing costs seconds and
    can fail on exactly the broken install this command is meant to describe.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:      # pragma: no cover - stdlib since 3.8
        return None, None
    try:
        v = version("torch")
    except PackageNotFoundError:
        return None, None
    m = _LOCAL.search(v)
    return v, (m.group("tag") if m else None)


def cuda_orphans(names=None):
    """Installed CUDA support packages that swapping torch alone would leave behind.

    Asked of the environment rather than hardcoded, so the uninstall line names what is actually
    on this machine and nothing else.
    """
    if names is None:
        try:
            from importlib.metadata import distributions
        except ImportError:      # pragma: no cover - stdlib since 3.8
            return []
        names = []
        for dist in distributions():
            name = (dist.metadata["Name"] or "") if dist.metadata else ""
            if name:
                names.append(name)
    return sorted({n for n in names
                   if n.lower().startswith("nvidia-") or n.lower() in ("triton", "pytorch-triton")})


def plan(*, system=None, machine=None, gpus=None, driver=None, torch_version=None, variant=None,
         compute=None):
    """What this machine should do, as (verdict, reason, pip arguments or None).

    Pure, and every input is injectable, so the table can be tested for machines nobody here owns.
    """
    system = system or platform.system()
    machine = machine or platform.machine()

    has_gpu = bool(gpus)
    cannot_tell = gpus is None

    # NO TORCH AT ALL is a different state from the wrong torch, and it is checked first because
    # every message below is about which BUILD is installed. Found while exercising an installed
    # wheel with `--no-deps`: the command reported "nothing is broken" to an environment that
    # cannot abliterate anything. torch is a base dependency now, so this means a damaged install
    # or a deliberate `--no-deps`, and either way the remedy is not a channel.
    if torch_version is None:
        reason = ("torch is not installed at all, so nothing here can run. It is a base "
                  "dependency, so this is a damaged install or one made with --no-deps: "
                  "`pip install --force-reinstall senbonzakura`, then run this again to get the "
                  "build your hardware can use.")
        return ("blocked", reason, None)

    def swap(channel):
        return ["install", "--upgrade", "--force-reinstall",
                "--index-url", f"{INDEX}/{channel}", "torch"]

    if system == "Darwin":
        # Metal needs Apple silicon. An Intel Mac was being told it uses Metal because this
        # branched on the operating system alone and read `machine` only in the final fallback.
        if machine not in ("arm64", "aarch64"):
            reason = (f"macOS on {machine}, which is an Intel Mac: there is no Metal backend and "
                      f"no CUDA build for it, so torch here runs on CPU. That is the only option "
                      f"on this hardware, not a fault to fix.")
            return ("ok", reason, None)
        reason = ("macOS on Apple silicon uses Metal and the default wheel is the right one; there "
                  "is no CUDA variant to choose.")
        return ("ok", reason, None)

    # An AMD build is neither the CUDA default nor a fault. This used to fall through to the Linux
    # branch and be told "the default Linux wheel is the CUDA build either way", which is a
    # statement about a wheel this machine is not running.
    if variant and variant.startswith("rocm"):
        reason = (f"torch is {torch_version}, an AMD ROCm build. This command only knows how to "
                  f"choose between PyTorch's CPU and CUDA channels, so it has no advice to give "
                  f"here; nothing about a ROCm install is wrong.")
        return ("ok", reason, None)

    if system == "Windows":
        if cannot_tell:
            reason = ("no nvidia-smi on PATH, so this cannot tell whether there is a card. On "
                      "Windows the default wheel is CPU-only, so if you do have an NVIDIA GPU it "
                      "is idle right now. Install the driver, or pass --cuda to choose a build.")
            return ("unknown", reason, None)
        if not has_gpu:
            return ("ok", "no NVIDIA GPU, and the Windows default wheel is CPU-only already.", None)
        if driver is None:
            # NOT "your driver is too old". Measured on the ROG: nvidia-smi there prints
            # "CUDA UMD Version", the regex read nothing, and the command told the operator to
            # update a driver that had run CUDA all night. A refusal for a reason that is not
            # true sends the reader to fix the wrong thing, which is the expensive kind of
            # correct-looking answer.
            reason = ("an NVIDIA GPU is present and the CUDA version this driver supports could "
                      "not be read, so the right channel cannot be chosen for you. torch here is "
                      "a CPU-only build, so the card is idle. Pass --cuda with a channel "
                      "(cu130, cu128, ...) from https://pytorch.org/get-started/locally/.")
            return ("unknown", reason, None)
        channel = pick_cuda_channel(driver, compute)
        if channel is None:
            if compute is not None and pick_cuda_channel(driver) is not None:
                reason = (f"an NVIDIA GPU is present and the driver is new enough (CUDA "
                          f"{driver[0]}.{driver[1]}), but the card is compute capability "
                          f"{compute[0]}.{compute[1]} and no channel this tool knows still ships "
                          f"kernels that old. Pass --cuda with an older channel, or use the CPU "
                          f"build deliberately.")
                return ("blocked", reason, None)
            reason = (f"an NVIDIA GPU is present and the driver reports CUDA "
                      f"{driver[0]}.{driver[1]}, which is older than any build PyTorch publishes. "
                      f"Update the driver, or pass --cuda to choose a channel yourself.")
            return ("blocked", reason, None)
        if variant and variant.startswith("cu"):
            return ("ok", f"already on a CUDA build ({torch_version}).", None)
        reason = (f"an NVIDIA GPU is present and torch is {torch_version or 'not installed'}, "
                  f"which on Windows is a CPU-only build. The card cannot be used until this is "
                  f"replaced.")
        return ("fix", reason, swap(channel))

    if system == "Linux":
        if has_gpu:
            if variant == "cpu":
                channel = pick_cuda_channel(driver, compute)
                if channel is None:
                    # NOT a guess. This used to fall back to `cu130` when the driver's ceiling was
                    # unreadable or below every channel, which hands a driver capped at 12.x a
                    # build it cannot run and produces the silent idle card this whole module
                    # exists to end. Windows already refused to guess in exactly this state; the
                    # two branches now give the same answer to the same question.
                    if driver is None:
                        reason = (f"an NVIDIA GPU is present and torch is {torch_version}, a "
                                  f"CPU-only build, so the card is idle. The CUDA version this "
                                  f"driver supports could not be read, so the right channel "
                                  f"cannot be chosen for you: pass --cuda with a channel (cu130, "
                                  f"cu128, ...) from https://pytorch.org/get-started/locally/.")
                        return ("unknown", reason, None)
                    too_old = (compute is not None and pick_cuda_channel(driver) is not None)
                    reason = (
                        (f"an NVIDIA GPU is present, the driver is new enough (CUDA "
                         f"{driver[0]}.{driver[1]}), but the card is compute capability "
                         f"{compute[0]}.{compute[1]} and no channel this tool knows still ships "
                         f"kernels that old. Pass --cuda with an older channel.")
                        if too_old else
                        (f"an NVIDIA GPU is present and the driver reports CUDA "
                         f"{driver[0]}.{driver[1]}, which is older than any build PyTorch "
                         f"publishes. Update the driver, or pass --cuda yourself."))
                    return ("blocked", reason, None)
                reason = (f"an NVIDIA GPU is present and torch is {torch_version}, a CPU-only "
                          f"build, so the card is idle.")
                return ("fix", reason, swap(channel))
            if compute is not None and compute < min(CHANNEL_ARCH_FLOOR.values()):
                reason = (f"an NVIDIA GPU is present and torch is {torch_version}, but the card is "
                          f"compute capability {compute[0]}.{compute[1]}, older than any channel "
                          f"this tool knows still ships kernels for. If runs fail with `no kernel "
                          f"image is available`, that is why.")
                return ("unknown", reason, None)
            reason = "the default Linux wheel is the CUDA build, and there is a card to use it."
            return ("ok", reason, None)
        if cannot_tell:
            # WHAT IS INSTALLED STILL MATTERS HERE. The first version of this branch said the
            # install was "larger than a machine with no card needs" without looking at the
            # install, and said it on a machine already carrying the CPU build. Not knowing
            # whether there is a card does not license a claim about what is on disk.
            if variant == "cpu":
                reason = (f"no nvidia-smi on PATH, so this cannot tell whether there is a card. "
                          f"torch is {torch_version}, a CPU-only build, so if there IS an NVIDIA "
                          f"GPU here it is idle. Install the driver and re-run, or pass --cuda.")
                return ("unknown", reason, None)
            reason = ("no nvidia-smi on PATH, so this cannot tell whether there is a card. The "
                      "default Linux wheel is the CUDA build either way, so torch itself is not "
                      "broken; it is simply larger than a machine with no card needs. If there IS "
                      "an NVIDIA GPU here, the driver is not installed and torch will not reach "
                      "it: install the driver and run this again.")
            return ("unknown", reason, None)
        if variant == "cpu":
            reason = f"no NVIDIA GPU, and torch is already the CPU build ({torch_version})."
            return ("ok", reason, None)
        # NOT "reclaims several gigabytes". The command below replaces torch and nothing else, and
        # pip has no autoremove, so the ~15 nvidia-* wheels and triton (which ARE the gigabytes)
        # stay on disk as orphans. `main` prints the second command that actually frees them.
        reason = ("no NVIDIA GPU, and the default Linux wheel drags 15 CUDA packages that cannot "
                  "be used here. Swapping torch for the CPU build is the first half; the CUDA "
                  "packages have to be removed separately, because pip never removes anything a "
                  "replaced package left behind.")
        return ("slim", reason, swap("cpu"))

    reason = f"no advice for {system}/{machine}; the default wheel is what pip chose."
    return ("unknown", reason, None)


def _describe(system, machine, gpus, driver, torch_version, variant, log, compute=None):
    log(f"  platform       {system} {machine}, Python {platform.python_version()}")
    if gpus is None:
        log("  NVIDIA driver  not found (nvidia-smi is not on PATH)")
    elif not gpus:
        log("  NVIDIA driver  present, and it reports no GPU")
    else:
        log(f"  NVIDIA driver  {len(gpus)} GPU(s): {'; '.join(g[:60] for g in gpus)}")
    if driver:
        log(f"  driver CUDA    up to {driver[0]}.{driver[1]}")
    if compute:
        log(f"  oldest card    compute capability {compute[0]}.{compute[1]}")
    log(f"  torch          {torch_version or 'not installed'}"
        + (f"  ({variant} build)" if variant else ""))


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="senbonzakura setup",
        description="Put the right build of torch on this machine. pip cannot do this itself: "
                    "there is no environment marker for a GPU, and PyPI cannot depend on "
                    "PyTorch's index.")
    p.add_argument("--apply", action="store_true",
                   help="actually run the pip command. Without this, nothing is changed and the "
                        "command is only printed.")
    p.add_argument("--cuda", metavar="CHANNEL",
                   help="force a PyTorch channel (cpu, cu128, cu130, ...) instead of the one the "
                        "driver implies. For a machine whose driver this cannot read.")
    a = p.parse_args(argv)

    gpus = nvidia_gpus()
    driver = driver_cuda_version()
    compute = compute_capability()
    torch_version, variant = installed_torch()
    system, machine = platform.system(), platform.machine()

    print("senbonzakura setup")
    _describe(system, machine, gpus, driver, torch_version, variant, print, compute=compute)

    if a.cuda and torch_version is not None:
        # `--cuda` bypasses the driver's answer on purpose, and it used to bypass the channel table
        # too, so a typo produced a confident install command against a URL that 404s.
        accepted, complaint = check_channel(a.cuda)
        if not accepted:
            print(f"\n  REFUSED: {complaint}", file=sys.stderr)
            return 2
        if complaint:
            print(f"\n  NOTE: {complaint}")
        too_old = card_is_too_old_for(a.cuda, compute)
        if too_old:
            print(f"\n  WARNING: {too_old}")
        verdict, reason = "fix", f"--cuda {a.cuda} was given, so the driver's own answer is not used."
        args = ["install", "--upgrade", "--force-reinstall",
                "--index-url", f"{INDEX}/{a.cuda}", "torch"]
    else:
        # `plan` checks "torch is not installed at all" first, and that answer outranks --cuda:
        # proposing a channel for a package that is not there sends the reader to the wrong fix.
        verdict, reason, args = plan(system=system, machine=machine, gpus=gpus, driver=driver,
                                     torch_version=torch_version, variant=variant, compute=compute)

    print(f"\n  {verdict.upper()}: {reason}")
    if args is None:
        # "blocked" means there is a real problem here that this command cannot fix, which is not
        # the same answer as "nothing to do". Exiting 0 on both makes them indistinguishable to
        # anything that gates on this, and `doctor` already sets the precedent of exiting non-zero
        # when the install cannot do its job.
        return 3 if verdict == "blocked" else 0

    printable = " ".join([sys.executable, "-m", "pip", *args])
    print(f"\n  {printable}")

    orphans = cuda_orphans() if verdict == "slim" else []
    if orphans:
        print(f"\n  and then, to actually free the space, because the line above replaces torch "
              f"and leaves these {len(orphans)} packages behind:\n")
        print("  " + " ".join([sys.executable, "-m", "pip", "uninstall", "-y", *orphans]))

    if not a.apply:
        print("\n  Nothing was changed. Re-run with --apply to do it.")
        return 0

    print("\n  running it...")
    try:
        rc = subprocess.run([sys.executable, "-m", "pip", *args], check=False,
                            timeout=PIP_TIMEOUT).returncode
    except subprocess.TimeoutExpired:
        print(f"\n  pip did not finish within {PIP_TIMEOUT}s and was stopped. torch may be "
              f"half-installed: run the command above by hand to see where it gets stuck.",
              file=sys.stderr)
        return 2
    except (OSError, subprocess.SubprocessError) as e:
        print(f"\n  FAILED to run pip: {e}", file=sys.stderr)
        return 2
    if rc != 0:
        print(f"\n  pip exited {rc}. Nothing here can fix that; the command above is the one to "
              f"debug.", file=sys.stderr)
        return rc
    print("\n  done. `senbonzakura doctor` will confirm what this install can now do.")
    return 0


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(main())
