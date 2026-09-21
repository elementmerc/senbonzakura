# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`tools/` is grouped by what a script is for, and a grouped script still finds the repository.

WHY THIS FILE EXISTS

`tools/` was a flat directory of thirty-seven scripts holding five different kinds of thing: hook
bodies, CI and release checkers, build and packaging, research diagnostics that need a GPU, and
developer conveniences. Nothing told a reader which was which. On 2026-09-21 they were grouped.

THE PART THAT WOULD HAVE BROKEN SILENTLY, and it is the reason this file is a test rather than a
note. Ten of those scripts locate the repository from their own position:

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

Moving a script one directory deeper makes that resolve to `tools/src`, which does not exist. AND
`sys.path.insert` OF A NON-EXISTENT PATH DOES NOT RAISE. The script keeps running and imports
`senbonzakura` from wherever else it can find one, which on a developer's machine is the installed
copy and in a clean container is nothing at all. So the failure is either invisible, because the
installed copy happens to agree with the tree, or it is a confusing ImportError a long way from
its cause. That is the same shape as every "works on my machine" defect this project has already
paid for, and a grep cannot see it because the path is built from components rather than written
out.

So the invariant is asserted directly: whatever root a script computes, that root must be this
repository. It is checked by looking for `pyproject.toml`, because that is what makes a directory
the root rather than a directory that happens to contain `src`.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

#: The groups, and what each one is for. A new script goes in one of these; a new group is a
#: deliberate decision rather than somewhere to put a file nobody could classify.
GROUPS = {
    "packaging": "produces something that ships: corpora, tracks, vendored binaries, wheels",
    "ci": "checks an artefact or a tree, and is run by CI or a release",
    "dev": "a convenience for somebody working on the tool, run by hand",
    "hooks": "the body of a git hook, wired in by a symlink under .githooks/",
    "research": "a diagnostic or experiment, usually needing a model and a GPU",
}

_ROOT_FROM_FILE = re.compile(r"Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]")


def _tool_scripts():
    return sorted(p for p in TOOLS.rglob("*")
                  if p.is_file() and p.suffix in (".py", ".sh")
                  and "__pycache__" not in p.parts)


def test_every_script_lives_in_a_group():
    """Nothing loose at the top level, or the grouping decays back to a flat directory."""
    loose = sorted(p.name for p in TOOLS.iterdir()
                   if p.is_file() and p.suffix in (".py", ".sh"))
    assert not loose, (
        f"{loose} sit directly in tools/ rather than in one of {sorted(GROUPS)}. "
        f"Pick the group it belongs to, or argue for a new one.")


def test_no_group_exists_that_is_not_described():
    """A directory nobody documented is the beginning of the flat directory coming back."""
    dirs = sorted(p.name for p in TOOLS.iterdir() if p.is_dir() and p.name != "__pycache__")
    assert dirs == sorted(GROUPS), f"groups on disk {dirs} do not match the documented {sorted(GROUPS)}"


@pytest.mark.parametrize("script", _tool_scripts(), ids=lambda p: str(p.relative_to(TOOLS)))
def test_a_script_that_locates_the_repository_finds_this_one(script):
    """THE INVARIANT THE MOVE COULD HAVE BROKEN IN SILENCE.

    Every `parents[N]` in a tools script is a claim about how deep it sits. The claim is checked
    against the filesystem rather than counted by eye, so moving a script between groups, or into
    a group one level deeper, fails here instead of resolving to a directory that does not exist
    and being inserted onto `sys.path` without complaint.
    """
    try:
        text = script.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        pytest.skip(f"not readable as text ({e.__class__.__name__})")
    wrong = []
    for m in _ROOT_FROM_FILE.finditer(text):
        computed = script.resolve().parents[int(m.group(1))]
        if not (computed / "pyproject.toml").is_file():
            wrong.append(f"parents[{m.group(1)}] resolves to {computed}, which is not the repository")
    assert not wrong, (
        f"{script.relative_to(ROOT)} computes a root that is not this repository: {wrong}. "
        f"`sys.path.insert` of a path that does not exist does NOT raise, so this would have "
        f"imported some other copy of the package, or none, without saying so.")


def test_the_hook_symlinks_point_at_real_files():
    """The wiring nothing else checks, because a symlink target is not text a grep can find.

    Both hook bodies live in `tools/hooks/` and are reached through symlinks under `.githooks/`.
    A grep for the filename finds nothing, which is how the audit that preceded this move nearly
    concluded one of them was unreferenced and deletable.
    """
    hooks = ROOT / ".githooks"
    if not hooks.is_dir():
        pytest.skip("no .githooks in this checkout, so there is no wiring to inspect")
    links = [p for p in hooks.rglob("*") if p.is_symlink()]
    if not links:
        pytest.skip("no hook symlinks in this checkout")
    broken = [f"{p.relative_to(ROOT)} -> {p.readlink()}" for p in links if not p.exists()]
    assert not broken, f"these hook symlinks do not resolve: {broken}"
