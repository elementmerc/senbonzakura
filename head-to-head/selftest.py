#!/usr/bin/env python3
"""Prove the sealed box is actually sealed, from inside it.

`run-isolated.sh` claims six invariants. A claim in a comment is not a control, and the flags that
enforce them are easy to lose one at a time: someone debugging a failing arm adds `-e HF_TOKEN`,
or drops `--network none` to let a tool fetch a tokeniser, and the box keeps its name while losing
its point.

So this runs INSIDE the container and checks each invariant by trying to violate it. Run it before
any arm, and after any change to the runner:

    head-to-head/run-isolated.sh --tool selftest --model <any dir> --out <dir> -- python /work/selftest.py

Exit 0 means every invariant held. Exit 1 names the ones that did not.
"""
import os
import socket
import sys

import torch

RESULTS = []


def check(label, got, want=True):
    """Record one invariant. Returns the result so a caller can branch on it."""
    good = (got == want)
    RESULTS.append((good, label, got))
    print(f"  {'PASS' if good else 'FAIL'}  {label}: {got}")
    return good


def refuses_write(path):
    """True when writing to `path` is refused. The point of a read-only mount."""
    try:
        with open(path, "w"):
            pass
    except OSError:
        return True
    # Reached only when the write SUCCEEDED, which is the failure. Clean up so a second run of this
    # test is not affected by the first one's leftovers.
    try:
        os.unlink(path)
    except OSError:
        pass
    return False


def _ruler_imports_alone():
    """The shared ruler must import inside the other tool's environment.

    `senbonzakura.metrics` is deliberately import-free so it can be read here without dragging in
    the abliterator, whose pinned transformers version is not the one this image holds. If that
    ever stops being true, the failure belongs here, before an arm spends hours on the card, rather
    than at the end of the run when the selection pass tries to score its candidates.
    """
    sys.path.insert(0, "/work/senbon-src")
    try:
        from senbonzakura.metrics import heretic_keyword_rate, knee_scalar
    except Exception as error:
        print(f"        import failed: {error}")
        return False
    # Exercised rather than merely imported, so a module that imports but misbehaves is caught too.
    return heretic_keyword_rate(["I cannot help with that"]) == 1.0 and knee_scalar(0, 0, 0, 0) == 0


def main():
    print("sealed-box self-test")

    # The GPU has to be reachable or the whole benchmark runs on CPU and its runtime column is
    # comparable with nothing. This is the invariant most likely to fail silently: with only
    # /usr/lib/wsl/lib mounted and not /usr/lib/wsl/drivers, libcuda loads and reports no devices.
    if check("cuda is available", torch.cuda.is_available()):
        x = torch.randn(512, 512, device="cuda")
        check("a matmul actually runs on the GPU", float((x @ x).abs().sum()) > 0)
        print(f"        device: {torch.cuda.get_device_name(0)}")

    check("the model mount exists", os.path.isdir("/model"))
    check("the model mount refuses writes", refuses_write("/model/.selftest-probe"))
    check("the root filesystem refuses writes", refuses_write("/usr/local/lib/.selftest-probe"))

    # Our own source is mounted when a pass inside the box needs to import our rulers. It is an
    # input like any other, so it is read-only: the tool under test must not be able to edit the
    # ruler it is about to be measured with. Checked only when the mount is present, because most
    # arms do not need it.
    if os.path.isdir("/work/senbon-src"):
        check("the senbonzakura source mount refuses writes",
              refuses_write("/work/senbon-src/.selftest-probe"))
        check("the shared ruler imports without pulling in the abliterator",
              _ruler_imports_alone())

    with open("/work/out/selftest-probe.txt", "w") as f:
        f.write("ok")
    check("the output directory accepts writes", os.path.exists("/work/out/selftest-probe.txt"))

    # The strongest control, so it is checked against a real connect rather than a config read.
    socket.setdefaulttimeout(3)
    try:
        socket.create_connection(("1.1.1.1", 53)).close()
        reachable = True
    except OSError:
        reachable = False
    check("the network is unreachable", not reachable)

    check("not running as root", os.getuid() != 0)
    print(f"        uid={os.getuid()}")

    # A credential reaching this box would defeat the point, so look for the usual ones by name.
    leaked = [v for v in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "AWS_SECRET_ACCESS_KEY",
                          "GITHUB_TOKEN", "RUNPOD_API_KEY") if os.environ.get(v)]
    check("no credential is present in the environment", leaked, [])

    failed = [label for good, label, _ in RESULTS if not good]
    if failed:
        print(f"\n  {len(failed)} INVARIANT(S) FAILED: {', '.join(failed)}")
        return 1
    print(f"\n  all {len(RESULTS)} invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
