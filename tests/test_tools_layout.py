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

#: THE SHELL IDIOM, and leaving it out cost a red CI run. The Python scripts were all corrected
#: when `tools/` was grouped; four SHELL scripts computing the same thing the same way were not,
#: because the pattern above reads Python only and the file list already included `.sh`. So the
#: guard walked straight past `ROOT="$(cd "$(dirname "$0")/.." && pwd)"` in four files, every one
#: of which then resolved to `tools/` instead of the repository. CI found it as
#: `cp: cannot stat .../tools/tools/ci/clean_room_checks.py`.
#:
#: The lesson is the one this project keeps relearning: a guard that covers one spelling of a
#: defect reports clean on the others, and reporting clean is worse than not running.
_ROOT_FROM_SHELL = re.compile(r'dirname "\$0"\)((?:/\.\.)+)')


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


@pytest.mark.parametrize("script", [p for p in _tool_scripts() if p.suffix == ".sh"],
                         ids=lambda p: str(p.relative_to(TOOLS)))
def test_a_shell_script_that_locates_the_repository_finds_this_one(script):
    """THE SAME INVARIANT AS ABOVE, IN THE OTHER LANGUAGE, and it was missing until CI failed.

    `ROOT="$(cd "$(dirname "$0")/.." && pwd)"` is the shell spelling of `parents[1]`. When these
    scripts moved one directory deeper it silently became `tools/`, so every `$ROOT/docs`,
    `$ROOT/.venv` and `$ROOT/dist-cleanroom` pointed at a path that does not exist. Unlike the
    Python case this one does not fail at import; it fails later, in a `cp` or a `python -m
    build`, a long way from the cause.

    Counted from the `..` segments rather than matched literally, so a script that walks three
    levels is checked against three levels rather than being excused for not matching a pattern.
    """
    try:
        text = script.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        pytest.skip(f"not readable as text ({e.__class__.__name__})")
    wrong = []
    for m in _ROOT_FROM_SHELL.finditer(text):
        levels = m.group(1).count("..")
        computed = script.resolve().parents[levels]
        if not (computed / "pyproject.toml").is_file():
            wrong.append(f"{levels} level(s) up resolves to {computed}, which is not the repository")
    assert not wrong, (
        f"{script.relative_to(ROOT)} computes a root that is not this repository: {wrong}. "
        f"Unlike the Python case this does not fail loudly at import; it fails later in a `cp` "
        f"or a build, a long way from its cause.")


def test_the_shell_pattern_actually_matches_the_idiom_it_guards():
    """A guard whose regex has gone stale reports every file clean.

    The Python half of this file has real call sites keeping it honest. The shell half would
    silently match nothing if the idiom were reformatted, so the pattern is asserted against the
    exact string that caused the outage.
    """
    sample = 'ROOT="$(cd "$(dirname "$0")/.." && pwd)"'
    m = _ROOT_FROM_SHELL.search(sample)
    assert m and m.group(1).count("..") == 1, sample
    deeper = 'ROOT="$(cd "$(dirname "$0")/../.." && pwd)"'
    m2 = _ROOT_FROM_SHELL.search(deeper)
    assert m2 and m2.group(1).count("..") == 2, deeper


# ── the spelling on disk is not the only spelling ───────────────────────────────────────────

#: Everything that names a `tools/` script from outside `tools/`. The two guards above read the
#: scripts themselves, which is the wrong end for this class: a script that moved is correct
#: where it now sits, and what breaks is somebody ELSE's string pointing at where it used to be.
#:
#: THE ONE THAT ESCAPED. `ci.yml` mounts `tools/` into a container as `/check` and ran
#: `/check/image_is_honest.py`, which became `tools/ci/image_is_honest.py` in the grouping. The
#: container job had been red since, and the error surfaced as a missing file inside a container
#: eleven lines below a `doctor` report full of expected failures, which is where nobody looks.
#: That is the fifth spelling of one move, after ten Python scripts, four shell scripts, the ruff
#: per-file table and two Dockerfiles.
_CALLERS = ("Dockerfile", "Dockerfile.cuda", ".github/workflows", "docs", "README.md",
            "CONTRIBUTING.md", "pyproject.toml", "Makefile")

#: A `tools/...` path written out in full, or the same path relative to a mount of `tools/`.
#: Both spellings, because the container one is the one that got away: `/check/ci/x.py` carries
#: no `tools/` prefix at all, so a pattern looking only for `tools/` reads the file, finds
#: nothing, and reports clean. That is this project's recurring failure and it is not repeated
#: here just because the fix is a second pattern.
_TOOLS_PATH = re.compile(r"(?<![\w/.-])tools/([\w./-]+\.(?:py|sh))")
_MOUNTED_PATH = re.compile(r"/check/([\w./-]+\.(?:py|sh))")


def _referenced_tools_paths():
    """Every `tools/` script named from outside `tools/`, as (where it was written, what it names)."""
    out = []
    for name in _CALLERS:
        target = ROOT / name
        files = sorted(p for p in target.rglob("*") if p.is_file()) if target.is_dir() else (
            [target] if target.is_file() else [])
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for pattern in (_TOOLS_PATH, _MOUNTED_PATH):
                for match in pattern.finditer(text):
                    out.append((path.relative_to(ROOT), match.group(1)))
    return out


def test_there_are_references_to_check():
    """Without this the parametrised test below silently becomes zero cases, which is the exact
    way the container path stayed broken: something examined the tree and asked nothing.
    """
    assert len(_referenced_tools_paths()) >= 5, (
        "no tools/ script is referenced from any workflow, Dockerfile or doc, which cannot be "
        "true while CI runs them")


@pytest.mark.parametrize("where,named", _referenced_tools_paths(),
                         ids=lambda v: str(v).replace("/", "-"))
def test_every_tools_script_named_from_outside_still_exists(where, named):
    assert (TOOLS / named).is_file(), (
        f"{where} names tools/{named}, which does not exist. Moving a script and leaving a "
        f"caller behind fails where the caller runs, which for a container mount is inside the "
        f"image and a long way from this repository.")
