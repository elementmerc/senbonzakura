#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""One job on the card at a time, with a deadline, and a reason when it refuses.

WHY THIS EXISTS, AND IT IS A MEASURED COST RATHER THAN A PRECAUTION

On 2026-10-05 two driver scripts ran against the same 6 GB card for over two hours. The log shows
both of them scoring `gemma-2-2b` at the same time. Four of that queue's eight failures were
`rc=124`, the 90 minute `timeout`, and all four fell inside exactly that window: six hours of card
time spent producing nothing, because neither script had any way to ask whether the other was
already using the card.

A lock file alone would not have caught it, and that is the part worth understanding.

THE CHECK THAT LOOKS RIGHT AND IS NOT

The obvious pre-flight is "is the card free", read as a memory threshold:

    nvidia-smi --query-gpu=memory.used    ->   1213 MiB   ->   under 1500, so go ahead

That passed on 2026-10-07 while a small model was actively training, and put a second job onto the
same card. A threshold cannot distinguish a card holding a finished process's cached allocator
blocks from a card holding a live job, because a small job simply does not use much memory.

**Occupancy is a process question.** `--query-compute-apps` lists the processes computing on the
device, and an empty list is the only reading that means nothing is running:

    nvidia-smi --query-compute-apps=pid,used_memory,process_name

Under WSL the `used_memory` column prints `[N/A]`, so a caller that parses the memory and gives up
when it cannot sees no processes and reports a free card. The presence of a ROW is the signal; the
numbers in it are extra. That is why this module tests rows and never values.

So there are two independent facts and this tool checks both: the lock says whether one of OUR
jobs claims the card, and the compute-app list says whether anything at all is on it. A lock with
nothing running is stale and is broken with a notice; a free lock with something running belongs to
somebody else, and we wait rather than pile on.

USAGE, from a shell driver

    # Exclusive, waits up to 20 minutes, runs the command, always releases
    tools/research/gpu_lock.py run --wait 1200 --label pass2 -- ./score_one.sh gemma-2-2b

    # Or inspect without taking anything
    tools/research/gpu_lock.py check

`run` is the form to prefer, because the lock is released even when the command crashes, is killed,
or the machine takes the lid down. The bare `acquire` form exists for a driver that cannot wrap its
own work, and such a driver must trap its exits itself.

Exit codes are the interface for a shell:

    0   the command ran (and its own status is passed through)
    4   the card is busy and the deadline passed
    5   the card could not be read at all, so nothing was claimed
"""
import argparse
import contextlib
import errno
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time

#: Where the lock lives. Not in `/tmp` on a machine whose `/tmp` is a tmpfs, and not in the repo,
#: because a lock is machine state rather than project state. `XDG_RUNTIME_DIR` is the correct home
#: and is cleaned on logout, which is exactly the lifetime a lock should have.
DEFAULT_LOCK = os.path.join(
    os.environ.get("XDG_RUNTIME_DIR") or os.path.expanduser("~/.cache"),
    "senbonzakura-gpu.lock")

#: Seconds to wait for the card by default. Zero would make this a check rather than a lock, and an
#: unbounded wait is the deadlock baseline section 2 forbids: every lock acquisition has a deadline.
DEFAULT_WAIT = 0.0

#: How often to re-ask while waiting. Long enough that a four hour wait is not thousands of
#: subprocesses, short enough that a freed card is picked up promptly.
POLL_SECONDS = 20.0

#: `nvidia-smi` is given a deadline too. A driver that hangs takes the query with it, and a
#: pre-flight that can hang forever is worse than no pre-flight.
SMI_TIMEOUT = 15.0


class CardUnreadableError(RuntimeError):
    """`nvidia-smi` is absent, failed, or timed out, so occupancy is unknown.

    UNKNOWN IS NOT FREE. Treating an unreadable card as free is how a second job lands on it, and
    it is the same error shape as reading a memory threshold: both answer a question that was not
    asked. The caller is told, and decides.
    """


def compute_apps(index=0, *, timeout=SMI_TIMEOUT):
    """The processes computing on the device, as a list of dicts. Empty means nothing is running.

    Raises `CardUnreadableError` rather than returning an empty list when it cannot ask, because those
    two answers must never be spelled the same way.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise CardUnreadableError("nvidia-smi is not on PATH, so what is on the card cannot be read.")
    try:
        out = subprocess.run(
            [exe, f"--id={int(index)}",
             "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as e:
        raise CardUnreadableError(
            f"nvidia-smi did not answer within {timeout:.0f}s. The driver may be wedged; "
            f"nothing was claimed.") from e
    except OSError as e:
        raise CardUnreadableError(f"nvidia-smi could not be run: {e}") from e
    if out.returncode != 0:
        raise CardUnreadableError(
            f"nvidia-smi exited {out.returncode}: {(out.stderr or '').strip() or 'no message'}")

    rows = []
    for raw in out.stdout.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("no running"):
            continue
        parts = [p.strip() for p in line.split(",")]
        # THE ROW IS THE SIGNAL AND THE FIELDS ARE EXTRA. Under WSL `used_memory` is `[N/A]`, so a
        # parse that insists on a number drops a live process and reports a free card.
        pid = parts[0] if parts else ""
        rows.append({
            "pid": int(pid) if pid.isdigit() else None,
            "process_name": parts[1] if len(parts) > 1 else "",
            "used_memory": parts[2] if len(parts) > 2 else "",
        })
    return rows


def pid_is_alive(pid):
    """Whether `pid` still exists. Used only to tell a stale lock from a held one."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive and owned by somebody else, which still counts as alive.
        return True
    except (OSError, ValueError):
        return False
    return True


def read_holder(path):
    """What the lock file says, or None when it says nothing readable.

    A lock whose contents cannot be parsed is treated as a lock with no holder recorded, not as a
    free lock: the flock underneath it is what actually excludes, and this is only for the message.
    """
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def describe_refusal(path, apps):
    """Why the card cannot be taken, phrased for a person rather than for a log scraper.

    `apps` is the compute-app list, or **None** when the card could not be read at all. Those are
    different facts and this never spells them the same way.
    """
    holder = read_holder(path)
    lines = []
    if holder:
        age = time.time() - float(holder.get("since", 0) or 0)
        alive = pid_is_alive(holder.get("pid"))
        lines.append(
            f"the lock is held by {holder.get('label') or 'an unlabelled job'} "
            f"(pid {holder.get('pid')}, {'running' if alive else 'GONE'}, "
            f"{age / 60:.0f} min ago): {holder.get('command') or 'no command recorded'}")
        if not alive:
            lines.append("  that pid is gone, so the lock is stale and the next acquire will "
                         "break it.")
    else:
        lines.append(f"the lock at {path} is held by a process that recorded no details.")
    if apps is None:
        # UNKNOWN IS NOT EMPTY, and writing it as empty here was this module making, in its own
        # refusal message, the exact error it exists to prevent. "Nothing is computing on the
        # card" is a claim; "I could not ask" is the fact when nvidia-smi cannot be run.
        lines.append("  what is on the card could not be read, so this says nothing about "
                     "whether the lock holder is still working.")
    elif apps:
        lines.extend(f"  on the card: pid {app['pid']} {app['process_name']} "
                     f"(memory {app['used_memory'] or 'not reported'})" for app in apps)
    else:
        lines.append("  nothing is computing on the card, so whoever holds the lock is between "
                     "jobs or has leaked it.")
    return "\n".join(lines)


@contextlib.contextmanager
def gpu_lock(path=DEFAULT_LOCK, *, wait=DEFAULT_WAIT, label="", index=0, log=print,
             require_idle=True):
    """Hold the card exclusively for the body, or raise before the body runs.

    DEADLINE BOUNDED, always. `wait` is seconds and zero means one attempt; there is deliberately
    no value meaning "forever", because an unbounded lock acquisition on a hot path is the hang
    baseline section 2 exists to forbid.

    `require_idle` also refuses when a process we do NOT own is computing on the card. That is the
    case the lock cannot see: a hand-run script, another user, or a job started before this tool
    existed. Set it False only when sharing is deliberate.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    deadline = time.monotonic() + max(0.0, float(wait))
    said_waiting = False
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                got = True
            except OSError as e:
                if e.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                got = False

            if got:
                # The flock is ours. Now the second, independent question: is anything on the card
                # that the lock could not know about?
                apps = compute_apps(index) if require_idle else []
                foreign = [a for a in apps if a["pid"] not in (None, os.getpid())]
                if not foreign:
                    break
                fcntl.flock(fd, fcntl.LOCK_UN)
                if not said_waiting:
                    log("gpu lock: the lock was free and the card is NOT. Waiting rather than "
                        "adding a second job to it.")
                    for a in foreign:
                        log(f"  pid {a['pid']} {a['process_name']} "
                            f"(memory {a['used_memory'] or 'not reported'})")
                    said_waiting = True
            elif not said_waiting:
                try:
                    apps = compute_apps(index)
                except CardUnreadableError:
                    apps = None
                log("gpu lock: the card is taken. " + describe_refusal(path, apps))
                said_waiting = True

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"the card was still busy after waiting {float(wait):.0f}s. Nothing was run. "
                    f"Raise --wait, or run this when the card is free; piling a second job onto "
                    f"one card is what this refusal exists to prevent, and it has cost this "
                    f"project six hours of card time once already.")
            time.sleep(min(POLL_SECONDS, remaining))

        os.truncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, json.dumps({
            "pid": os.getpid(),
            "label": label,
            "command": " ".join(sys.argv[1:]) or "(none recorded)",
            "since": time.time(),
            "host": os.uname().nodename,
        }).encode("utf-8"))
        os.fsync(fd)
        log(f"gpu lock: held by pid {os.getpid()}"
            + (f" for {label}" if label else "") + f", lock at {path}")
        try:
            yield path
        finally:
            # RELEASED ON EVERY PATH, including a crash and a kill that lets finally run. A lock
            # that outlives its job turns one wasted slot into every slot after it.
            with contextlib.suppress(OSError):
                os.truncate(fd, 0)
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            log("gpu lock: released")
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


def build_parser():
    ap = argparse.ArgumentParser(
        prog="gpu_lock.py",
        description="One job on the card at a time. Checks the lock AND what is computing on the "
                    "device, because a memory threshold passes while a small job trains.")
    ap.add_argument("mode", choices=["run", "check"],
                    help="'run' holds the lock for a command and always releases it; 'check' "
                         "reports what holds the card and takes nothing")
    ap.add_argument("--lock", default=DEFAULT_LOCK, help=f"lock file (default: {DEFAULT_LOCK})")
    ap.add_argument("--wait", type=float, default=DEFAULT_WAIT,
                    help="seconds to wait for the card (default: 0, one attempt). There is no "
                         "value meaning forever")
    ap.add_argument("--label", default="", help="a name for this job, recorded in the lock so a "
                                                "refusal can say who holds it")
    ap.add_argument("--index", type=int, default=0, help="which GPU (default: 0)")
    ap.add_argument("--share-ok", action="store_true",
                    help="take the lock even when another process is computing on the card. Off "
                         "by default, and the default is the lesson: two drivers on one 6 GB card "
                         "wasted four 90 minute slots")
    # NOT `argparse.REMAINDER`, and this is a defect this file shipped for one test run.
    # REMAINDER is greedy from the first positional, so `run --label x --share-ok -- cmd` parsed
    # `--share-ok` as part of the COMMAND and the flag silently did nothing: the one flag whose
    # whole job is to override a refusal, quietly dead. `split_argv` below does the split itself,
    # which is deterministic and leaves every option on the left where argparse can see it.
    return ap


def split_argv(argv):
    """`(options, command)`, split on the first bare `--`.

    Done by hand rather than with `argparse.REMAINDER`, for the reason recorded above.
    """
    argv = list(argv)
    if "--" not in argv:
        return argv, []
    cut = argv.index("--")
    return argv[:cut], argv[cut + 1:]


def main(argv=None):
    options, cmd = split_argv(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(options)

    if a.mode == "check":
        try:
            apps = compute_apps(a.index)
        except CardUnreadableError as e:
            print(f"gpu lock: {e}")
            return 5
        if not os.path.exists(a.lock):
            print(f"gpu lock: no lock file at {a.lock}.")
        else:
            print(describe_refusal(a.lock, apps))
        if not apps:
            print("gpu lock: nothing is computing on the card.")
        return 0

    if not cmd:
        raise SystemExit("gpu_lock.py run needs a command: `run --label x -- ./script.sh`.")

    try:
        with gpu_lock(a.lock, wait=a.wait, label=a.label, index=a.index,
                      require_idle=not a.share_ok):
            return subprocess.run(cmd, check=False).returncode
    except TimeoutError as e:
        print(f"gpu lock: {e}")
        return 4
    except CardUnreadableError as e:
        # UNREADABLE IS NOT FREE, so this refuses rather than running. A driver that would rather
        # proceed blind can pass --share-ok, which is a decision somebody has to type.
        print(f"gpu lock: {e} Nothing was run. Pass --share-ok to proceed without this check.")
        return 5


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
