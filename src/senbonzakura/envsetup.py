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
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out.stdout or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def pick_cuda_channel(driver):
    """The newest channel this driver can carry, or None when it can carry none of them."""
    if driver is None:
        return None
    return next((tag for need, tag in CUDA_CHANNELS if need <= driver), None)


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


def plan(*, system=None, machine=None, gpus=None, driver=None, torch_version=None, variant=None):
    """What this machine should do, as (verdict, reason, pip arguments or None).

    Pure, and every input is injectable, so the table can be tested for machines nobody here owns.
    """
    system = system or platform.system()
    machine = machine or platform.machine()

    has_gpu = bool(gpus)
    cannot_tell = gpus is None

    def swap(channel):
        return ["install", "--upgrade", "--force-reinstall",
                "--index-url", f"{INDEX}/{channel}", "torch"]

    if system == "Darwin":
        reason = ("macOS builds use Metal and the default wheel is the right one; there is no "
                  "CUDA variant to choose.")
        return ("ok", reason, None)

    if system == "Windows":
        if cannot_tell:
            reason = ("no nvidia-smi on PATH, so this cannot tell whether there is a card. On "
                      "Windows the default wheel is CPU-only, so if you do have an NVIDIA GPU it "
                      "is idle right now. Install the driver, or pass --cuda to choose a build.")
            return ("unknown", reason, None)
        if not has_gpu:
            return ("ok", "no NVIDIA GPU, and the Windows default wheel is CPU-only already.", None)
        channel = pick_cuda_channel(driver)
        if channel is None:
            seen = f"{driver[0]}.{driver[1]}" if driver else "an unreadable version"
            reason = (f"an NVIDIA GPU is present and the driver reports CUDA {seen}, which is "
                      f"older than any build PyTorch publishes. Update the driver, or pass --cuda "
                      f"to choose a channel yourself.")
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
                reason = (f"an NVIDIA GPU is present and torch is {torch_version}, a CPU-only "
                          f"build, so the card is idle.")
                return ("fix", reason, swap(pick_cuda_channel(driver) or "cu130"))
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
                      "default Linux wheel is the CUDA build either way, so nothing is broken; it "
                      "is simply larger than a machine with no card needs.")
            return ("unknown", reason, None)
        if variant == "cpu":
            reason = f"no NVIDIA GPU, and torch is already the CPU build ({torch_version})."
            return ("ok", reason, None)
        reason = ("no NVIDIA GPU, and the default Linux wheel drags 15 CUDA packages that cannot "
                  "be used here. Replacing it with the CPU build reclaims several gigabytes.")
        return ("slim", reason, swap("cpu"))

    reason = f"no advice for {system}/{machine}; the default wheel is what pip chose."
    return ("unknown", reason, None)


def _describe(system, machine, gpus, driver, torch_version, variant, log):
    log(f"  platform       {system} {machine}, Python {platform.python_version()}")
    if gpus is None:
        log("  NVIDIA driver  not found (nvidia-smi is not on PATH)")
    elif not gpus:
        log("  NVIDIA driver  present, and it reports no GPU")
    else:
        log(f"  NVIDIA driver  {len(gpus)} GPU(s): {'; '.join(g[:60] for g in gpus)}")
    if driver:
        log(f"  driver CUDA    up to {driver[0]}.{driver[1]}")
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
    torch_version, variant = installed_torch()
    system, machine = platform.system(), platform.machine()

    print("senbonzakura setup")
    _describe(system, machine, gpus, driver, torch_version, variant, print)

    if a.cuda:
        verdict, reason = "fix", f"--cuda {a.cuda} was given, so the driver's own answer is not used."
        args = ["install", "--upgrade", "--force-reinstall",
                "--index-url", f"{INDEX}/{a.cuda}", "torch"]
    else:
        verdict, reason, args = plan(system=system, machine=machine, gpus=gpus, driver=driver,
                                     torch_version=torch_version, variant=variant)

    print(f"\n  {verdict.upper()}: {reason}")
    if args is None:
        return 0

    printable = " ".join([sys.executable, "-m", "pip", *args])
    print(f"\n  {printable}")
    if not a.apply:
        print("\n  Nothing was changed. Re-run with --apply to do it.")
        return 0

    print("\n  running it...")
    try:
        rc = subprocess.run([sys.executable, "-m", "pip", *args], check=False).returncode
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
