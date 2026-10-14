# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Nothing tracked in this repository names the machines it was built on.

This repository has a public remote. Everything git tracks is published, including
the comments, the docstrings and the test names, and those are where private detail
leaks: not through a credential, which the commit hooks already scan for, but through
a reader slowly learning the shape of somebody's network from thirty incidental
mentions of where a bug was found.

The finding that produced this file, 2026-09-17: twenty-nine tracked files named
machines, a job runner and home paths. None of them was a secret on its own. Together
they described a fleet. A comment saying a defect was found on a particular box
carries no more information for the reader than one saying it was found on a Windows
laptop with a card in it, and the second is the useful sentence anyway, because it
says what about that machine mattered.

WHAT REPLACES A NAME. Describe the machine by what it has. "A CPU-only box", "a
Windows laptop", "a 6 GB card", "a second machine", "a peer session". Those are what
the reader needs to judge whether the measurement applies to them, which a nickname
never told them.

WHY THE TERMS ARE NOT IN THIS FILE, which is the part worth reading twice. A list of
the names that must not be published is a list of the names, and committing it to a
public repository publishes every one of them with a label explaining why they matter.
The check is tracked; what it looks for is not. The terms live in an untracked local
file, and when that file is absent the check says so loudly and skips rather than
passing quietly, because a guard that reports success having inspected nothing is the
failure this project has now hit four separate times.

The consequence, stated plainly: this check does NOT run in CI, because CI has no
copy of the list and must not be given one. It runs on a machine that has the list,
which is the same machine the leak would originate from. That is the whole enforceable
surface, and it is better than the alternative of a public denylist.
"""
import re
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent

#: One term per line, `#` for comments, each line a regular expression. Lives under the
#: private tree, which `.git/info/exclude` keeps out of every commit, so the list itself
#: is never published. See this module's docstring for why that is not optional.
TERMS_FILE = _ROOT / "private" / "infrastructure-terms.txt"

#: There is deliberately no exemption list. The first version of this file carried one,
#: for the operator tooling that was tracked despite living behind `.git/info/exclude`,
#: and the honest fix turned out to be untracking that tooling rather than teaching the
#: check to look away from it. If any of it comes back, this should fail.

#: THE HALF THAT NEEDS NO PRIVATE DATA, added 2026-10-14 after the operator asked that this
#: class of leak be made impossible rather than merely caught here.
#:
#: WHY IT WAS NEEDED. The project-specific half reads its terms from a file in `private/`,
#: which is correct: a public list of the names we are hiding would be the leak itself. But
#: `private/` is a symlink into an unversioned tree, so it is ABSENT from every worktree and
#: every fresh clone, and the check skipped there. A skip reads exactly like a pass in a tail
#: of output, so the guard was dead in most of the places it ran. Two agents reported it
#: skipping on the same day, and a home-directory path reached this repository's public
#: history in July under the same conditions.
#:
#: A home directory is structurally recognisable WITHOUT knowing whose it is, which is what
#: makes this half shippable in a public file: the pattern names no account, it describes a
#: shape. So this half always runs, everywhere, and the private list only ever ADDS to it.
HOME_PATHS = (
    (re.compile(r"/home/[A-Za-z0-9._-]+/"), "a Linux home directory, which names an account"),
    (re.compile(r"/Users/[A-Za-z0-9._-]+/"), "a macOS home directory, which names an account"),
    (re.compile(r"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9._-]+"), "a Windows profile, which names an account"),
    (re.compile(r"\bssh://[A-Za-z0-9._-]+@"), "an ssh URL carrying a username"),
)

#: Account names that name NOBODY. The property this guard protects is "no tracked file names a
#: real account", and documentation has to be able to show the shape of a path without inventing
#: a person. Exempting by ACCOUNT rather than by FILE is deliberate: a file allowlist would hide
#: every future finding in that file, which is where a real leak goes to live. This list stays
#: short, and a name is added to it only when it obviously belongs to no one.
PLACEHOLDER_ACCOUNTS = frozenset({
    "someone", "somebody", "user", "username", "builduser", "youruser", "me", "you", "example",
})

#: Paths that legitimately contain one of the shapes above. Kept tiny and exact, because an
#: allowlist is where a real finding goes to hide. Each entry says why it is here.
HOME_PATH_ALLOWED = {
    # The guard's own patterns and the prose explaining them.
    "tests/test_no_private_infrastructure_is_named.py": "this file defines the patterns",
}


def _account_in(text):
    """The account component of the first home-ish path in `text`, or None."""
    for rx, _why in HOME_PATHS:
        m = rx.search(text)
        if not m:
            continue
        # The account is the last non-empty path or userinfo component of the match.
        parts = [p for p in re.split(r"[/\\@:]", m.group(0)) if p]
        if parts:
            return parts[-1]
    return None


def _load_terms():
    if not TERMS_FILE.is_file():
        return None
    out = []
    for raw in TERMS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _tracked_text_files():
    # NOT check=True. The suite is deliberately run from a `git archive` extract and inside
    # containers, where this is not a checkout and git exits non-zero. Raising there turns a
    # condition this check cannot inspect into a suite error, which is noise rather than a
    # finding; the empty result makes the fixture skip and say why. Found on the build box,
    # 2026-09-17, the first time this file ran anywhere but the machine that wrote it.
    try:
        proc = subprocess.run(
            ["git", "-C", str(_ROOT), "ls-files", "-z"],
            capture_output=True, text=True, check=False)
    except OSError:  # git absent entirely
        return
    if proc.returncode != 0:
        return
    out = proc.stdout
    here = Path(__file__).name
    for name in out.split("\0"):
        if not name or Path(name).name == here:
            continue
        p = _ROOT / name
        if not p.is_file():
            continue
        try:
            yield name, p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


@pytest.fixture(scope="module")
def terms():
    loaded = _load_terms()
    if loaded is None:
        pytest.skip(
            f"no term list at {TERMS_FILE.relative_to(_ROOT)}, so this check inspected "
            "NOTHING. It is deliberately not committed; create it locally, one regular "
            "expression per line, to arm this guard on this machine.")
    if not loaded:
        pytest.fail("the term list exists and is empty, which arms nothing")
    return loaded


@pytest.fixture(scope="module")
def tracked():
    files = list(_tracked_text_files())
    if not files:
        pytest.skip("not a git checkout, so there is nothing tracked to inspect")
    return files


def test_no_tracked_file_names_private_infrastructure(tracked, terms):
    """Every tracked line, against every term. One failure listing all of them.

    Reported as one assertion rather than one per term on purpose: the reader wants the
    whole surface in front of them when they start a sweep, not a test run that reveals
    the next name each time they fix the last one.
    """
    rx = re.compile("|".join(f"(?:{t})" for t in terms), re.IGNORECASE)
    hits = []
    for name, text in tracked:
        for n, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{name}:{n}: {line.strip()[:100]}")
    assert not hits, (
        f"{len(hits)} tracked line(s) name private infrastructure, and this repository "
        "is public. Describe the machine by what it has, not by what it is called:\n  "
        + "\n  ".join(hits[:20])
        + ("" if len(hits) <= 20 else f"\n  ... and {len(hits) - 20} more"))


def test_the_committed_ignore_file_does_not_advertise_the_private_trees(terms):
    """A public `.gitignore` naming a private directory tells every visitor it exists.

    The per-clone `.git/info/exclude` is where those rules belong. Same reasoning as the
    rest of this file, one layer down: the rule keeps the tree out, and where the rule is
    written decides whether the tree's name goes out.
    """
    gitignore = _ROOT / ".gitignore"
    if not gitignore.is_file():
        pytest.skip("no committed ignore file")
    rx = re.compile("|".join(f"(?:{t})" for t in terms), re.IGNORECASE)
    named = [
        line for line in gitignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and rx.search(line)
    ]
    assert not named, (
        "the committed ignore file names a private tree: " + "; ".join(named))


def test_no_tracked_file_names_a_home_directory():
    """The half that cannot skip, because it needs nothing from `private/`.

    THIS IS THE ONE THAT WOULD HAVE CAUGHT THE JULY LEAK. Two evidence files recorded the
    absolute path of the dataset they scored, which named an account, and they reached the
    public remote. The project-specific guard beside this one could have caught it and did
    not run, because its term list lives in a tree that a worktree and a fresh clone do not
    have, and an absent list made it skip.

    So this asserts the property that does not depend on knowing any of our names. A reader
    gains nothing from an account name in a path, and a stranger gains a little, which is the
    whole of the argument for removing it.
    """
    files = list(_tracked_text_files())
    if not files:
        pytest.skip("not a git checkout, so there is nothing tracked to inspect")

    hits = []
    # The helper already yields (name, text) and has done the reading, the decoding and the
    # binary filtering. Re-opening each path here would have duplicated all three, and the
    # first draft of this test did exactly that and failed on the tuple.
    for rel, body in files:
        if rel in HOME_PATH_ALLOWED:
            continue
        for number, line in enumerate(body.splitlines(), start=1):
            for rx, why in HOME_PATHS:
                if not rx.search(line):
                    continue
                account = _account_in(line)
                if account and account.lower() in PLACEHOLDER_ACCOUNTS:
                    break  # a documented shape, naming nobody
                hits.append(f"{rel}:{number}: {why}: {line.strip()[:110]}")
                break

    assert not hits, (
        f"{len(hits)} tracked line(s) name a home directory, and this repository is public. "
        f"Describe the location by what it holds, not by whose account it is under:\n    "
        + "\n    ".join(sorted(hits)[:25]))


def test_the_home_directory_patterns_actually_match():
    """The patterns are checked against examples, so an empty result means absence not blindness.

    A regex that matches nothing passes the test above on every repository in the world. This
    is what tells the two apart, and it is the reason that test can be trusted when it is quiet.
    """
    should_match = (
        "/home/someone/models/x",
        "  model: /Users/someone/Downloads/ckpt",
        r"C:\\Users\\someone\\models",
        "ssh://someone@example.invalid:2222/repo.git",
    )
    for sample in should_match:
        assert any(rx.search(sample) for rx, _ in HOME_PATHS), f"no pattern matched {sample!r}"

    should_not_match = (
        "/home/",                      # no account after it
        "/usr/share/doc/thing",
        "the home directory of the process",
        "https://example.invalid/a/b",
    )
    for sample in should_not_match:
        assert not any(rx.search(sample) for rx, _ in HOME_PATHS), f"a pattern matched {sample!r}"
