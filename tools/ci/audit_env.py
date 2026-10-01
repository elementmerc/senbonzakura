#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Audit an already-installed environment for known advisories, and refuse to look clean.

WHY THIS AUDITS THE ENVIRONMENT RATHER THAN A REQUIREMENTS FILE

The plan this comes from said "pip-audit against each of the three resolved environments", and
the obvious reading, a job that reconstructs each environment from its constraints file, does
not work. Measured 2026-10-01, all three ways:

    constraints/ci.txt      pins `torch==2.13.0+cpu`, a local version segment that exists on the
                            PyTorch CPU index and not on PyPI, so the resolve fails outright.
    constraints/floors.txt  pins `pyarrow==14`, which has no cp314 wheel, so on a 3.14 runner pip
                            tries to build it from source and fails. It resolves on 3.10, which
                            is the interpreter the floor job already uses.
    the floating set        `pip install --dry-run` over `.[dev]` did not finish in five minutes.

Each failure is specific to the environment, and each environment ALREADY EXISTS, correctly
built, in the job that builds it. So this runs as a step inside those jobs and audits what was
installed, which costs nothing extra and is a stronger claim: it audits the environment rather
than a reconstruction of it that might differ.

WHY THE PACKAGE COUNT IS ASSERTED, WHICH IS THE PART THAT MATTERS

Measured 2026-10-01 on the development machine: `pip list` reported 206 installed packages,
`pip-audit --strict` audited 28 of them and printed "No known vulnerabilities found" with exit
status 0. The 28 were one `site-packages` tree and the 178 it skipped were another. **`--strict`
did not catch it**: that flag fails the audit when dependency COLLECTION fails, not when
collection quietly returns a subset.

So an audit that covered 14% of the environment and an audit that covered all of it both exit 0
with the same sentence. That is this project's recurring defect, and the count floor below is
the same assertion the lockfile job makes with `-eq 12` and the workflow lint makes with its file
count. A gate whose exit status cannot distinguish "clean" from "barely looked" is decoration.

THE THREE OUTCOMES, because two is how a check becomes decoration

    PASS         the advisory service answered, the expected surface was covered, nothing found
                 (or only entries the allowlist names, and the count of those is PRINTED).
    FAIL         advisories were found, or the covered surface was smaller than required, or
                 dependency collection failed under --strict.
    DID NOT RUN  the advisory service was unreachable. Loud, and NOT red, because a gate that
                 goes red on the network teaches people to re-run until it is green.

The caller probes reachability before invoking this, so by the time this runs the network is
known good and no failure here has to be guessed at from an error string.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def accepted_advisories() -> tuple[list[str], dict[str, dict]]:
    """The one allowlist, read from pyproject so CI and the pre-push gate cannot disagree.

    `[tool.invariant] audit_ignore` is what the pre-push gate passes to pip-audit.
    `[tool.senbonzakura.security] accepted_advisories` is its machine-readable twin, carrying the
    assessed version and the mitigation. Two files naming two different sets of accepted risk is
    the drift this reads one file to avoid.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_bytes().decode("utf-8"))
    ignore = list(data.get("tool", {}).get("invariant", {}).get("audit_ignore", []))
    detail = {
        entry["id"]: entry
        for entry in data.get("tool", {})
        .get("senbonzakura", {})
        .get("security", {})
        .get("accepted_advisories", [])
    }
    # Every ignored id must carry its assessment. An id in `audit_ignore` with no entry beside it
    # is an accepted risk nobody wrote down, which is the shape that becomes permanent.
    undocumented = sorted(set(ignore) - set(detail))
    if undocumented:
        sys.exit(
            f"::error::{', '.join(undocumented)} appear in [tool.invariant] audit_ignore with no "
            f"matching entry in [tool.senbonzakura.security] accepted_advisories. An ignored "
            f"advisory with no written assessment is a hole nobody is forced to revisit."
        )
    return ignore, detail


def service_reachable(timeout: float = 20.0) -> bool:
    """Probe the advisory service before auditing, so no failure after this has to be guessed at.

    This is the pattern the lockfile job established and the reason it matters here: without a
    probe, telling "the index is down" from "the audit found something" means matching strings in
    pip-audit's stderr, and a gate whose verdict depends on an error message's wording breaks
    silently when the wording changes. Probing first makes every later failure a real one.

    It probes pypi.org because that is pip-audit's default vulnerability service. Switching the
    service with `-s osv` would need this endpoint changed to match, which is why the two are not
    separated.
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen("https://pypi.org/simple/", timeout=timeout) as response:
            return 200 <= response.status < 400
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True,
                    help="which resolved environment this is, for the report")
    ap.add_argument("--min-packages", type=int, required=True,
                    help="the floor on packages audited. A subset audit must not read as clean.")
    ap.add_argument("--report", action="store_true",
                    help="report without failing. The scheduled run uses this; the push path "
                         "never does, because an advisory published overnight should not turn a "
                         "green main red at 03:00 with nobody to act.")
    ap.add_argument("--skip-probe", action="store_true",
                    help="for testing this script's own guards offline")
    args = ap.parse_args(argv)

    # DID NOT RUN, and deliberately not red. An unreachable advisory service is a network result,
    # and a gate that goes red on the network teaches people to re-run until it is green, which
    # is how red stops meaning anything. It is loud instead, and it says plainly that nothing was
    # checked rather than letting a green tick imply it was.
    if not args.skip_probe and not service_reachable():
        print(
            f"::warning::DID NOT RUN: pypi.org is unreachable, so '{args.label}' was NOT audited "
            f"for advisories. This is a network result, not a pass."
        )
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as handle:
                handle.write(
                    f"### Dependency audit DID NOT RUN: {args.label}\n"
                    f"`pypi.org` was unreachable, so no advisory check happened for this "
                    f"environment in this run.\n"
                )
        return 0

    ignore, detail = accepted_advisories()

    # The console script first, the module second. pip-audit installs as both and which one is
    # importable depends on how it was installed: a `pip install --user` puts the script on PATH
    # while leaving the module off a system interpreter's sys.path, which is the case on the
    # development machine. A missing tool is a loud failure naming it, never a silent green.
    import shutil
    executable = shutil.which("pip-audit")
    if executable:
        cmd = [executable]
    elif importlib.util.find_spec("pip_audit") is not None:
        cmd = [sys.executable, "-m", "pip_audit"]
    else:
        print(
            "::error::DID NOT RUN: pip-audit is neither on PATH nor importable, so no advisory "
            "check happened for this environment. Install it in the job before this step. An "
            "absent scanner must never be reported as a clean scan."
        )
        return 1
    cmd += ["--strict", "--format", "json", "--progress-spinner", "off"]
    for advisory in ignore:
        cmd += ["--ignore-vuln", advisory]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    # pip-audit writes JSON to stdout and its diagnostics to stderr. An unparseable stdout means
    # the tool itself failed, which is DID NOT RUN and never a pass, whatever the exit status
    # was. This is the defect that bit me measuring actionlint: a wrong flag made it exit 2
    # printing usage, and a naive parse read that as zero findings and therefore as clean.
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(proc.stderr.strip()[-2000:], file=sys.stderr)
        print(
            f"::error::DID NOT RUN: pip-audit produced no parseable JSON for '{args.label}' "
            f"(exit {proc.returncode}). The tool failed rather than finding the environment "
            f"clean, and an unparseable result must never read as a pass."
        )
        return 1

    deps = payload.get("dependencies", [])
    findings = [(d["name"], d["version"], v) for d in deps for v in d.get("vulns", [])]

    print(f"environment: {args.label}")
    print(f"packages audited: {len(deps)} (floor {args.min_packages})")
    # The ignored count is printed rather than swallowed, so "clean" and "clean because we
    # accepted something" are different sentences. Verified 2026-10-01: a single
    # `--ignore-vuln PYSEC-2026-3804` turns exit 1 into exit 0 and prints "1 ignored" on the pypi
    # service and "2 ignored" on osv, the second because that advisory has an aliased duplicate
    # record. Both are suppressed by the one id, so the allowlist does not need the GHSA spelling.
    if ignore:
        print(f"advisories the allowlist accepts: {len(ignore)}")
        for advisory in ignore:
            entry = detail[advisory]
            print(f"  {advisory}  {entry['package']}=={entry['assessed_version']}")
            print(f"      mitigation: {entry['mitigation']}")
            print(f"      revisit:    {entry['revisit']}")

    # THE COVERAGE FLOOR. See the module docstring: --strict does not catch a subset audit.
    if len(deps) < args.min_packages:
        print(
            f"::error::pip-audit covered {len(deps)} packages in '{args.label}' and the floor is "
            f"{args.min_packages}. An audit of part of an environment exits 0 with the same "
            f"sentence as an audit of all of it, so the count is the only thing that can tell "
            f"them apart. Either the environment is not installed yet, or pip-audit scoped itself "
            f"to a subset of it. This is NOT a clean result."
        )
        return 1

    if findings:
        for name, version, vuln in findings:
            fix = ", ".join(vuln.get("fix_versions") or []) or "none published"
            print(f"  {name}=={version}  {vuln['id']}  fix: {fix}")
        verb = "::warning::" if args.report else "::error::"
        print(
            f"{verb}{len(findings)} advisory finding(s) in '{args.label}'. Each needs an "
            f"assessment: upgrade, mitigate and record it in "
            f"[tool.senbonzakura.security] accepted_advisories, or establish it does not apply. "
            f"Do not add an id to audit_ignore without the entry beside it."
        )
        return 0 if args.report else 1

    print(f"PASS: {len(deps)} packages audited, no advisories outside the allowlist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
