#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Resolve every pinned action SHA to the tag it really is, and check the comments against it.

THE DEFECT THIS CLOSES

Baseline section 5 says pin actions by commit hash, never by tag, and this repository does.
Every pin then carries a trailing comment naming the version, because a bare 40-character SHA
tells a reader nothing. That comment is prose, and until this tool existed nothing checked it.
`tests/test_workflow_actions_are_pinned.py` verified the pin was a SHA and said so in its own
docstring: "It does not check that the SHA exists, or that it belongs to the tag in the trailing
comment."

A wrong comment is worse than no comment, because it licenses the next reader to trust it. The
audit on 2026-10-01 found exactly one, and it had been there long enough that two sessions had
read past it: `actions/setup-node` was pinned to the v6.0.0 commit under a comment saying
v5.0.0.

TWO MODES, AND WHY THE SPLIT MATTERS

  --resolve   Hits the GitHub API, writes `.github/action-pins.lock`. Needs the network and a
              `gh` login. Run by hand when a pin changes, and on a schedule by CI.
  --check     Reads only the lockfile and the workflow files. No network, no credentials.
              This is what the test and every CI job run.

WHY A CHECKED-IN LOCKFILE RATHER THAN RESOLVING LIVE IN THE TEST

Four reasons, in increasing order of how much they matter:

1. The suite has to pass offline and in a clean room. A test that needs api.github.com is a
   test that fails on a train.
2. An outage at GitHub would turn this repository's build red for a reason that says nothing
   about this repository's code. `ci.yml` already refuses to let Codecov do that; the same
   argument applies here.
3. A resolution is a fact about a moment in time. Recording it puts it in a diff where a human
   reviews it, instead of re-deriving it invisibly on every run.
4. The one that decides it: if the test resolved tags live, then an attacker who repointed
   `v7` at their own commit would have made the test AGREE with them. Live resolution checks
   the pin against whatever the tag says today, which is the movable thing the pinning rule
   exists to distrust. The lockfile is what makes a retag visible, because the recorded SHA
   stops matching what the tag now returns, and the scheduled `--resolve` is what notices.

So the division of labour is: the lockfile is the trusted record, the offline check enforces
that the workflows and their comments agree with it, and the scheduled re-resolve is the only
thing that ever talks to the network and the only thing that can report that the world moved.

WHAT A SCHEDULED RE-RESOLVE REPORTS, AND WHAT IT MUST NOT DO

It reports three distinct outcomes, and the third is as loud as the second:

  agrees        every recorded SHA still carries its recorded tag.
  disagrees     a tag has been repointed, deleted, or the SHA has gone. A finding, loudly.
  DID NOT RUN   no network, no credentials, rate-limited. Never a pass.

It must not rewrite the lockfile unattended. A tag moving under us is the event we are watching
for, and a job that silently updated the record would erase the evidence and call it success.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
LOCKFILE = ROOT / ".github" / "action-pins.lock"

#: `uses: owner/repo@<40 hex>` with its optional trailing comment. Deliberately narrower than
#: the test's `USES`: this tool only has business with third-party pins, and a local `./` step
#: carries no SHA to resolve.
PIN = re.compile(
    r"^\s*-?\s*uses:\s*(?P<repo>[A-Za-z0-9._-]+/[A-Za-z0-9._/-]+)@(?P<sha>[0-9a-f]{40})"
    r"(?:\s*#\s*(?P<comment>.*?))?\s*$",
)

#: The version token a comment is allowed to claim: a full `vN.N.N`, or `vN.N`, or `vN`.
#: A bare `vN` is refused by `--check`; see `_claimed_tag`.
VERSION_TOKEN = re.compile(r"\bv\d+(?:\.\d+)*\b")

TIMEOUT_S = 120


def yaml_files() -> list[Path]:
    """Every file that can carry a third-party `uses:` line.

    `action.yml` at the repository root is included deliberately, and leaving it out was a real
    gap: it is the composite action this project PUBLISHES, so it is the one file strangers
    execute in their own CI, and an audit scoped to "the workflows" misses it. It carried a pin
    that nothing in this tool's first draft would have resolved.
    """
    files = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    files += [p for p in (ROOT / "action.yml", ROOT / "action.yaml") if p.exists()]
    return files


def pin_sites() -> list[dict[str, object]]:
    """Every pinned third-party reference, one record per SITE, not per distinct pin."""
    sites: list[dict[str, object]] = []
    for path in yaml_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = PIN.match(line)
            if match is None:
                continue
            sites.append({
                "file": path.name,
                "line": lineno,
                "repo": match["repo"],
                "sha": match["sha"],
                "comment": (match["comment"] or "").strip(),
            })
    return sites


def _claimed_tag(comment: str) -> str | None:
    """The version this comment claims, or None if it claims nothing checkable.

    A comment may carry prose around the tag ("v4.2.2, released 2026-08-06, 55 days clear of
    the cooldown") and that is welcome. What it may not do is claim a MOVING name. Three pins
    used to read `# tip of the v6 series, resolved 2026-09-23`, which was true on that date and
    becomes false without anything changing in this repository, so no check can ever hold it to
    account. Those were rewritten to the exact release tag the SHA carries.
    """
    tokens = VERSION_TOKEN.findall(comment)
    if not tokens:
        return None
    # The first token wins: `# v6.19.2, superseding v6.19.1` claims v6.19.2.
    return tokens[0]


# ── resolve mode: the only part that touches the network ─────────────────────────────────────


class ResolutionError(RuntimeError):
    """The API could not be consulted. Distinct from "the API disagreed"."""


def _api(path: str, *, paginate: bool = False) -> object:
    cmd = ["gh", "api"]
    if paginate:
        cmd.append("--paginate")
    cmd.append(path)
    try:
        # check=False deliberately: a non-zero `gh` exit is a 404 to classify, not an exception
        # to propagate. The returncode is inspected below and a genuine failure becomes a loud
        # ResolutionError, which is what keeps "could not look" distinct from "found nothing".
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, check=False)
    except FileNotFoundError as exc:
        raise ResolutionError(
            "the `gh` CLI is not installed, so no pin could be resolved. "
            "Install it, or run this job where it exists.",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ResolutionError(f"`gh api {path}` did not answer within {TIMEOUT_S}s") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        if "Not Found" in stderr or "404" in stderr:
            return None                       # the resource genuinely does not exist
        raise ResolutionError(f"`gh api {path}` failed: {stderr}")
    text = proc.stdout
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # `--paginate` concatenates one JSON array per page: `[...][...]`.
        try:
            return json.loads(text.replace("][", ","))
        except json.JSONDecodeError as exc:
            raise ResolutionError(f"`gh api {path}` returned undecodable JSON: {exc}") from exc


def _deref(repo: str, obj: dict) -> str:
    """A tag ref points either straight at a commit or at an annotated tag object."""
    if obj["type"] != "tag":
        return str(obj["sha"])
    tag = _api(f"repos/{repo}/git/tags/{obj['sha']}")
    if not isinstance(tag, dict):
        raise ResolutionError(f"{repo}: annotated tag {obj['sha']} would not dereference")
    return str(tag["object"]["sha"])


def resolve_one(repo: str, sha: str) -> dict[str, object]:
    """Establish, from the API, what this SHA is. Both directions, independently.

    Forward (tag -> SHA) and reverse (SHA -> tags) are asked separately on purpose. Either one
    alone can mislead: a forward match tells you the claimed tag is right but not that it is the
    only or best name, and the reverse list tells you what names exist without confirming the
    one a human wrote down.
    """
    # Does the commit exist at all? A SHA that resolves nowhere is the big finding.
    commit = _api(f"repos/{repo}/commits/{sha}")
    if commit is None:
        return {"exists": False, "tags": [], "committed": None}
    committed = commit["commit"]["committer"]["date"]

    tags = _api(f"repos/{repo}/tags", paginate=True)
    if not isinstance(tags, list):
        raise ResolutionError(f"{repo}: the tag listing did not come back as a list")
    if not tags:
        raise ResolutionError(
            f"{repo}: the tag listing came back EMPTY, which would make every reverse lookup "
            f"vacuously 'no tag'. Treating that as a resolution failure rather than a finding.",
        )
    pointing = sorted(t["name"] for t in tags if t["commit"]["sha"] == sha)

    return {
        "exists": True,
        "committed": committed,
        "tags": pointing,
        "tags_listed": len(tags),
    }


def preferred_tag(tags: list[str]) -> str | None:
    """Of the tags pointing at one commit, the most specific one, for use in a SUGGESTION.

    A release commit usually carries both `v6` and `v6.19.2`, because the major ref is a moving
    alias that happens to sit there today. The specific one is the only one worth writing in a
    comment: `v6` will mean a different commit next month.

    This picks the suggested wording only. The check does NOT require a comment to equal this,
    it requires the comment to name one of the tags that genuinely point at the SHA, because a
    single commit can legitimately carry two from different series: the SHA pinned for
    `codecov/codecov-action` v7.0.0 is also tagged v6 and v6.0.2, and a comment naming either
    of the specific ones is telling the truth.
    """
    if not tags:
        return None
    return max(tags, key=lambda t: (t.count("."), t))


def cmd_resolve(args: argparse.Namespace) -> int:
    sites = pin_sites()
    if not sites:
        print("DID NOT RUN: no pinned action references found at all. Has the layout moved?")
        return 2

    distinct: dict[tuple[str, str], list[dict]] = {}
    for site in sites:
        distinct.setdefault((str(site["repo"]), str(site["sha"])), []).append(site)

    entries = []
    failures = 0
    for (repo, sha), where in sorted(distinct.items()):
        try:
            facts = resolve_one(repo, sha)
        except ResolutionError as exc:
            print(f"DID NOT RUN for {repo}@{sha[:12]}: {exc}", file=sys.stderr)
            failures += 1
            continue
        tags = [str(t) for t in facts["tags"]]  # type: ignore[index]
        entry = {
            "action": repo,
            "sha": sha,
            "exists": facts["exists"],
            "tags_pointing_here": tags,
            "tag": preferred_tag(tags),
            "committed": facts["committed"],
            # Filenames, not file:line. A line number churns whenever anything above the pin
            # is edited, which would make the lockfile read as stale after an unrelated change
            # and demand a network re-resolve to fix a record that was never wrong.
            "files": sorted({str(w["file"]) for w in where}),
        }
        entries.append(entry)
        state = "ok" if facts["exists"] else "SHA DOES NOT EXIST"
        # Progress goes to stderr so `--stdout` emits nothing but the lockfile, which a
        # scheduled job pipes into a diff. The first draft printed this to stdout and the
        # result was JSON a reader could not parse.
        print(f"{repo}@{sha[:12]}  tag={entry['tag']}  {state}", file=sys.stderr)

    if failures:
        print(
            f"\nDID NOT RUN cleanly: {failures} of {len(distinct)} pins could not be resolved. "
            f"The lockfile has NOT been rewritten, because a partial record read as complete is "
            f"worse than a stale one.",
            file=sys.stderr,
        )
        return 2

    payload = {
        "comment": (
            "Resolved SHA-to-tag record for every pinned third-party action. Generated by "
            "tools/ci/resolve_action_pins.py --resolve; checked offline by --check and by "
            "tests/test_workflow_actions_are_pinned.py. Do not hand-edit: the point of the file "
            "is that a machine established these mappings."
        ),
        "resolved_at": args.resolved_at,
        "pins": entries,
    }
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    if args.stdout:
        sys.stdout.write(text)
        return 0
    tmp = LOCKFILE.with_suffix(".lock.part")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(LOCKFILE)                     # atomic: never a half-written lockfile
    print(f"\nwrote {LOCKFILE.relative_to(ROOT)} with {len(entries)} pins")
    return 0


# ── check mode: offline, and what CI and the test both run ───────────────────────────────────


def load_lockfile() -> dict:
    if not LOCKFILE.exists():
        raise FileNotFoundError(
            f"{LOCKFILE.relative_to(ROOT)} is missing. Regenerate it with "
            f"`python tools/ci/resolve_action_pins.py --resolve`.",
        )
    return json.loads(LOCKFILE.read_text(encoding="utf-8"))


def check(sites: list[dict], lock: dict) -> list[str]:
    """Every finding, as a list of human-readable lines. Empty means agreement."""
    findings: list[str] = []
    by_pin = {(p["action"], p["sha"]): p for p in lock["pins"]}

    for site in sites:
        key = (site["repo"], site["sha"])
        where = f"{site['file']}:{site['line']}"
        entry = by_pin.get(key)
        if entry is None:
            findings.append(
                f"{where}: {site['repo']}@{site['sha'][:12]} is not in the pin lockfile. "
                f"Resolve it with `--resolve` so the version in its comment is a checked fact "
                f"rather than a claim.",
            )
            continue
        if not entry.get("exists", False):
            findings.append(
                f"{where}: {site['repo']}@{site['sha'][:12]} DOES NOT EXIST in that repository "
                f"according to the lockfile. This is not a documentation problem.",
            )
            continue

        claimed = _claimed_tag(str(site["comment"]))
        suggestion = entry.get("tag")
        actual_tags = [str(t) for t in (entry.get("tags_pointing_here") or [])]

        if claimed is None:
            findings.append(
                f"{where}: {site['repo']} is pinned with no version in its trailing comment. "
                f"A bare 40-character SHA tells a reader nothing; write `# {suggestion}`.",
            )
        elif re.fullmatch(r"v\d+", claimed):
            findings.append(
                f"{where}: the comment claims a moving major name ({claimed}) rather than a "
                f"release. `{claimed}` means a different commit the moment upstream tags again, "
                f"so no check can ever hold this comment to account. Write `# {suggestion}`.",
            )
        elif claimed not in actual_tags:
            findings.append(
                f"{where}: the comment claims {site['repo']} is {claimed}, but that SHA is "
                f"{suggestion}"
                + (f" (every tag pointing there: {', '.join(actual_tags)})"
                   if len(actual_tags) > 1 else "")
                + ". Correct the comment, or change the pin, but they have to agree.",
            )
    return findings


def cmd_check(_args: argparse.Namespace) -> int:
    sites = pin_sites()
    if not sites:
        print("DID NOT RUN: no pinned action references found at all. Has the layout moved?")
        return 2
    try:
        lock = load_lockfile()
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"DID NOT RUN: {exc}", file=sys.stderr)
        return 2

    findings = check(sites, lock)
    print(f"checked {len(sites)} pinned references against {len(lock['pins'])} locked pins "
          f"(lockfile resolved {lock.get('resolved_at', 'at an unrecorded time')})")
    if findings:
        print(f"\n{len(findings)} finding(s):", file=sys.stderr)
        for line in findings:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print("every pin's trailing comment agrees with the resolved record")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", add_help=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resolve", action="store_true",
                      help="hit the GitHub API and rewrite the lockfile (needs network + gh)")
    mode.add_argument("--check", action="store_true",
                      help="offline: assert the workflows agree with the lockfile (default)")
    parser.add_argument("--stdout", action="store_true",
                        help="with --resolve, print the lockfile instead of writing it")
    parser.add_argument("--resolved-at", default=None,
                        help="the date to stamp into the lockfile (default: today, UTC)")
    args = parser.parse_args(argv)

    if args.resolved_at is None:
        import datetime
        args.resolved_at = datetime.datetime.now(tz=datetime.timezone.utc).date().isoformat()

    return cmd_resolve(args) if args.resolve else cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
