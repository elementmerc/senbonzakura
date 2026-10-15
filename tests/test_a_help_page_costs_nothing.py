# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Printing a page of text must not load a deep-learning stack, and that is asserted, not hoped.

WHAT WAS WRONG, MEASURED ON A SIXTEEN-CORE WORKSTATION, 2026-10-10

`entry.py` exists so that `senbonzakura --help` does not import torch, and it works: the top level
answers in 0.07 seconds. Five of the delegated commands did not inherit it, because they reached
their parser helper through `.cli`, which imports torch, transformers and optuna at module scope.
Cold process, best of three:

    drift        3.568s      coherence    3.208s      compass      3.141s
    score        3.336s      validate     3.152s

Against `capability` and `jailbreak` at 0.05s, which already took the fix and say so in their own
source. The cost breaks down as torch 1.17s, transformers 0.79s, optuna 0.19s, the rest ours.

AFTER: score 0.075s, coherence 0.058s, drift 0.060s, validate 0.051s, compass 1.248s.

WHY THE TEST ASKS ABOUT MODULES AND NOT ABOUT SECONDS

A wall-clock assertion on a shared machine is a flaky test, and this project's own rule is that a
test which passes nine times in ten is not green. What actually matters is categorical: did
printing a help page load the deep-learning stack. That has one answer on every machine, it is
what the seconds were a proxy for, and it cannot drift with the hardware.

WHY A SUBPROCESS PER COMMAND

`sys.modules` is process state. Once any test in this suite has imported torch, an in-process
check can never see the difference again, which would make this guard report clean from the moment
it ran second. One cold process per command is the only honest version.

THE FAILURE THIS GUARDS AGAINST IS SUBTLE AND I WALKED INTO IT

A deferred import placed at the top of `main` buys nothing, because `--help` and every argument
error exit from inside `parse_args`, which runs after it. With the import one line too high, the
help page still cost the full three seconds, the code looked correct, and the only way to tell was
to ask which modules were loaded when it exited. So the thing being asserted here is not "the
imports are deferred", it is "the help page did not pay", and those are different claims.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from senbonzakura.entry import DELEGATED, typed_commands

#: The modules whose presence means a help page paid for the measurement stack.
HEAVY = ("torch", "transformers", "optuna", "datasets", "accelerate")

#: Commands whose `--help` legitimately loads torch, each with the reason.
#:
#: ONE ENTRY, AND IT IS A BLOCKER RATHER THAN A CHOICE. `margin.py` decorates two functions
#: `@torch.no_grad()`, and a decorator is evaluated when the module is imported, so the name has to
#: exist at module scope. Getting `compass` to where the other four are means either a deferring
#: `no_grad` decorator in a shared home or restructuring both bodies into `with` blocks, which is a
#: shared decision rather than a local fix. It is in DEFERRED.md; until then this is a ratchet, and
#: the list may only shrink.
MAY_PAY: dict[str, str] = {
    "compass": "margin.py decorates two functions @torch.no_grad(), and a decorator is evaluated "
               "at import time, so torch has to be at module scope until that is restructured",
}

#: What a probe prints. Kept as a constant so the two tests below drive one probe rather than two
#: that could disagree about what they measured.
_PROBE = r"""
import contextlib
import io
import json
import sys

from senbonzakura import entry

buffer = io.StringIO()
try:
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        entry.main([{name!r}, "--help"])
except SystemExit:
    pass
print(json.dumps({{
    "heavy": [m for m in {heavy!r} if m in sys.modules],
    "printed": len(buffer.getvalue()),
}}))
"""


def _probe(name):
    """What one command's `--help` loaded, in a process that has imported nothing else of ours."""
    done = subprocess.run(
        [sys.executable, "-c", _PROBE.format(name=name, heavy=HEAVY)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, check=False, timeout=600)
    lines = [ln for ln in done.stdout.strip().split("\n") if ln.startswith("{")]
    if not lines:
        pytest.fail(f"the probe for `{name} --help` printed no verdict.\n"
                    f"  stdout: {done.stdout[-500:]!r}\n  stderr: {done.stderr[-800:]!r}")
    return json.loads(lines[-1])


@pytest.mark.parametrize("name", sorted(typed_commands()))
def test_a_help_page_does_not_load_the_measurement_stack(name):
    result = _probe(name)
    assert result["printed"] > 0, (
        f"`senbonzakura {name} --help` printed nothing, so this test measured the cost of a page "
        f"nobody got")
    paid = result["heavy"]
    if name in MAY_PAY:
        return
    assert not paid, (
        f"`senbonzakura {name} --help` loaded {', '.join(paid)} to print a page of text. On a "
        f"development machine "
        f"that is about three seconds for torch, transformers and optuna together, and the reader "
        f"paying it is somebody trying to find out what the command takes.\n"
        f"  The fix is the one `capability.build_parser` writes out: reach `loader_parser` through "
        f"`.parser` rather than through `.cli`, and import the heavy names inside the functions "
        f"that use them.\n"
        f"  AND BELOW THE PARSE. An import at the top of `main` is still paid by `--help`, which "
        f"exits from inside `parse_args`. That mistake looks correct and measures identically to "
        f"no fix at all.\n"
        f"  If this command genuinely cannot avoid it, record it in MAY_PAY with the reason.")


@pytest.mark.parametrize("name", sorted(MAY_PAY))
def test_a_command_that_pays_still_pays_so_the_exemption_is_not_stale(name):
    """An exemption for a command that no longer needs one is an exemption covering the next one.

    The same reasoning as `test_every_exemption_names_a_flag_that_exists` in the unbounded-count
    file: a record nobody checks has stopped being a record. When `compass` is fixed, this fails and
    the entry comes out, which is the ratchet.
    """
    assert _probe(name)["heavy"], (
        f"`{name}` is in MAY_PAY and its help page no longer loads anything heavy. Remove the "
        f"entry: a stale exemption silently covers the next command that regresses into it.")


def test_the_exemption_list_only_shrinks():
    """A ratchet. Adding a name here is how the class quietly reopens, so the count is pinned."""
    assert len(MAY_PAY) <= 1, (
        f"MAY_PAY has grown to {len(MAY_PAY)}: a command's help page started loading the "
        f"measurement stack and was excused rather than fixed.")


def test_every_exemption_names_a_command_that_exists():
    unknown = sorted(set(MAY_PAY) - set(DELEGATED))
    assert not unknown, f"MAY_PAY names commands this package does not dispatch: {unknown}"


def test_the_probe_can_see_a_heavy_import_when_there_is_one():
    """A guard that cannot see what it guards reports clean, which is this project's through-line.

    So the probe is driven against a process that deliberately imports torch, and has to notice.
    Skipped where torch is absent, because then there is nothing to notice and saying so is
    honest; the assertion above still holds on such a machine, it is simply easier to satisfy.
    """
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed here, so a heavy import cannot be planted")
    source = _PROBE.format(name="gate", heavy=HEAVY).replace(
        "from senbonzakura import entry", "import torch  # planted\nfrom senbonzakura import entry")
    done = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, check=False, timeout=600)
    lines = [ln for ln in done.stdout.strip().split("\n") if ln.startswith("{")]
    assert lines, done.stderr[-500:]
    assert "torch" in json.loads(lines[-1])["heavy"], (
        "the probe did not notice a torch import it was handed deliberately, so every clean "
        "result above means nothing")
