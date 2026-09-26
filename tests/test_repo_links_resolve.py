# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The other half of `tools/ci/readme_links_resolve.py`: links back into the repository.

WHY THIS FILE EXISTS

A claim in this project is supposed to trace to a committed artefact, and the public route to an
artefact is a `github.com/elementmerc/senbonzakura/blob/<ref>/<path>` link. On 2026-09-25 every
one of those links named `main`, and the published `main` was 611 commits behind `dev` and
carried none of the artefacts that carry weight. Seven links, all 404, on the docs site, in the
issue-template menu and in the Colab notebook.

The checker is tested here rather than only run, and it is tested against real git refs made in a
temporary repository, because the specific way this hid was a LOCAL `main` that had been
fast-forwarded and never pushed. That clone answers yes to every path while a reader's browser
answers no, so a checker that asks the local branch would have reported the whole set as fine.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "repo_links_resolve",
    Path(__file__).resolve().parent.parent / "tools" / "ci" / "readme_links_resolve.py")
links = importlib.util.module_from_spec(_spec)
sys.modules["repo_links_resolve"] = links
_spec.loader.exec_module(links)

REPO = links.REPO
ROOT = Path(__file__).resolve().parent.parent


# ── the parser ───────────────────────────────────────────────────────────────────────

def test_all_three_url_shapes_are_read():
    """`blob` for a file, `tree` for a directory, `raw` for the bytes. All three 404 alike."""
    body = (f"[a]({REPO}/blob/main/METHOD.md)\n"
            f"[b]({REPO}/tree/dev/head-to-head/results/2026-09-10)\n"
            f"[c]({REPO}/raw/v0.4.0/icon.svg)\n")
    assert links.repo_links_in(body) == [
        ("main", "METHOD.md"),
        ("dev", "head-to-head/results/2026-09-10"),
        ("v0.4.0", "icon.svg"),
    ]


def test_a_fragment_a_query_and_a_full_stop_are_not_part_of_the_path():
    body = (f"see {REPO}/blob/dev/CONTRIBUTING.md#reporting-a-security-issue and "
            f"{REPO}/blob/dev/METHOD.md?plain=1 and then {REPO}/blob/dev/README.md.\n")
    assert links.repo_links_in(body) == [
        ("dev", "CONTRIBUTING.md"), ("dev", "METHOD.md"), ("dev", "README.md")]


def test_a_repeated_link_is_reported_once():
    body = f"{REPO}/blob/dev/METHOD.md twice: {REPO}/blob/dev/METHOD.md\n"
    assert links.repo_links_in(body) == [("dev", "METHOD.md")]


def test_a_link_to_the_repository_root_carries_no_path_to_check():
    """`.../senbonzakura` and `.../tree/dev` name no file, so there is nothing to resolve."""
    assert links.repo_links_in(f"[home]({REPO})\n[branch]({REPO}/tree/dev)\n") == []


# ── the published ref, which is the part that hid the defect ─────────────────────────

def _run(cwd, *args):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def clone(tmp_path):
    """A repository whose local `main` carries a file that its `origin/main` does not.

    That is the exact state of this project's checkout: `main` fast-forwarded locally, never
    pushed, 611 commits ahead of the branch the public sees.
    """
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _run(upstream, "git", "init", "-q", "-b", "main")
    _run(upstream, "git", "config", "user.email", "t@example.invalid")
    _run(upstream, "git", "config", "user.name", "Test")
    (upstream / "OLD.md").write_text("old\n", encoding="utf-8")
    _run(upstream, "git", "add", "OLD.md")
    _run(upstream, "git", "commit", "-q", "-m", "old")

    work = tmp_path / "work"
    _run(tmp_path, "git", "clone", "-q", str(upstream), str(work))
    _run(work, "git", "config", "user.email", "t@example.invalid")
    _run(work, "git", "config", "user.name", "Test")
    (work / "NEW.md").write_text("new\n", encoding="utf-8")
    _run(work, "git", "add", "NEW.md")
    _run(work, "git", "commit", "-q", "-m", "new, and not pushed")
    return work


def test_the_published_ref_is_preferred_over_the_local_branch(clone):
    assert links.published_ref(clone, "main", allow_fetch=False) == "origin/main"


def test_a_file_that_exists_only_locally_is_reported_broken(clone):
    """The whole point. `NEW.md` is in the working checkout and not in what a reader can fetch."""
    (clone / "docs.md").write_text(f"[new]({REPO}/blob/main/NEW.md)\n", encoding="utf-8")
    _run(clone, "git", "add", "docs.md")
    checked, broken, unresolvable = links.check_repo_links(clone, allow_fetch=False)
    assert checked == 1
    assert not unresolvable
    assert broken == [("docs.md", "main", "NEW.md")]


def test_a_file_the_published_ref_carries_passes(clone):
    (clone / "docs.md").write_text(f"[old]({REPO}/blob/main/OLD.md)\n", encoding="utf-8")
    _run(clone, "git", "add", "docs.md")
    checked, broken, unresolvable = links.check_repo_links(clone, allow_fetch=False)
    assert (checked, broken, unresolvable) == (1, [], {})


def test_a_ref_this_clone_cannot_answer_for_is_not_silently_a_pass(clone):
    """An unknown ref has to be reported, not skipped: a skip is a pass nobody asked for."""
    (clone / "docs.md").write_text(f"[x]({REPO}/blob/no-such-ref/OLD.md)\n", encoding="utf-8")
    _run(clone, "git", "add", "docs.md")
    checked, broken, unresolvable = links.check_repo_links(clone, allow_fetch=False)
    assert checked == 1
    assert not broken
    assert set(unresolvable) == {"no-such-ref"}


def test_only_tracked_prose_is_read(clone):
    """An untracked scratch file is not a surface a reader meets, and must not fail the build."""
    (clone / "scratch.md").write_text(f"[new]({REPO}/blob/main/NEW.md)\n", encoding="utf-8")
    checked, broken, _ = links.check_repo_links(clone, allow_fetch=False)
    assert (checked, broken) == (0, [])


# ── this repository, as it stands ────────────────────────────────────────────────────

def _has_history():
    """Whether this tree has git history, which is a stronger requirement than having the files.

    `check_repo_links` asks git to resolve the ref each link names and then to show whether the
    tree behind it carries the path. A released tarball, and the CI job that runs the suite from
    one, have every file and no history, so every ref is unresolvable and the scan legitimately
    reads nothing. That is an absent measurement rather than a broken link, and the two must not
    share a representation.
    """
    try:
        subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", "HEAD"],
                       capture_output=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def _remote_reachable():
    """Whether `origin` answers, so an unresolvable ref can be told from an unreachable network.

    Without this the test cannot distinguish "this link is broken" from "this machine is offline",
    and those must not share a representation: the first is a defect a reader meets and the second
    is a measurement that did not happen.
    """
    try:
        done = subprocess.run(["git", "-C", str(ROOT), "ls-remote", "--exit-code", "origin", "HEAD"],
                              capture_output=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


@pytest.mark.skipif(not _has_history(),
                    reason="no git history here, so no ref a link names can be resolved")
def test_the_notebook_and_the_issue_menu_name_a_ref_that_carries_their_files():
    """The two surfaces a reader meets before they have cloned anything.

    Scoped to the files this change owns. The docs-site pages are checked by the same function in
    CI; naming them here as well would make this test fail for somebody else's edit.
    """
    owned = ("notebooks/senbonzakura_colab.ipynb", ".github/ISSUE_TEMPLATE/config.yml")
    # FETCHING IS ALLOWED HERE, and it is the difference between a real check and a false failure.
    #
    # `actions/checkout` fetches only the ref that triggered the run. These two surfaces link to
    # `dev`, so a run triggered by a push to `dev` resolves it and a run triggered by the push to
    # `main` does not. On 2026-09-26 the same commit went green for the dev push and red for the
    # main promotion minutes later, which reads as flakiness and is not: it is this test asking a
    # single-branch checkout about a branch it was never given.
    #
    # `published_ref` already knows how to shallow-fetch the one ref it needs. The unit tests above
    # keep `allow_fetch=False` because they are about the function; this test is about the real
    # repository, where the alternative to fetching is reporting a broken link that is not broken.
    checked, broken, unresolvable = links.check_repo_links(ROOT, allow_fetch=True)
    assert checked >= 4, "the scan found almost nothing, so the pattern has gone quiet"
    mine = [b for b in broken if b[0] in owned]
    assert not mine, f"these link to a path the ref they name does not carry: {mine}"
    stray = {r for r, names in unresolvable.items() if names & set(owned)}
    if stray and not _remote_reachable():
        pytest.skip(f"cannot reach the remote, so {sorted(stray)} could not be resolved; that is "
                    f"an absent measurement rather than a broken link")
    assert not stray, f"a ref named by {owned} could not be resolved: {unresolvable}"
