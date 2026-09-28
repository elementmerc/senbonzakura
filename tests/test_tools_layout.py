# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
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
import subprocess
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
                out.extend((path.relative_to(ROOT), m.group(1))
                           for m in pattern.finditer(text))
    return out


def test_there_are_references_to_check():
    """Without this the parametrised test below silently becomes zero cases, which is the exact
    way the container path stayed broken: something examined the tree and asked nothing.
    """
    assert len(_referenced_tools_paths()) >= 5, (
        "no tools/ script is referenced from any workflow, Dockerfile or doc, which cannot be "
        "true while CI runs them")


@pytest.mark.parametrize(("where", "named"), _referenced_tools_paths(),
                         ids=lambda v: str(v).replace("/", "-"))
def test_every_tools_script_named_from_outside_still_exists(where, named):
    assert (TOOLS / named).is_file(), (
        f"{where} names tools/{named}, which does not exist. Moving a script and leaving a "
        f"caller behind fails where the caller runs, which for a container mount is inside the "
        f"image and a long way from this repository.")


# ── the half of the `ci` definition that nothing enforced ─────────────────────────────────────────

#: Where a `tools/ci` script can legitimately be invoked from. A script in that directory claims, by
#: the `GROUPS` table above, to be "run by CI or a release". These are the places that can make that
#: claim true. Being imported by a test is NOT one of them: a test proves the script works, not that
#: anything runs it.
_INVOKERS = (
    Path(".github") / "workflows",       # CI
    Path("RELEASING.md"),                # the release checklist a human follows
    # `.githooks` WAS HERE AND IS NOT TRACKED EITHER, found by the test below the moment it existed.
    # It is the operator's local hook directory, excluded from this repository, so it vouched for
    # scripts on one machine exactly as `scripts/runpod` did. `tools/hooks` stays, because the hook
    # BODIES are tracked and a clone can read them; what is untracked is the wiring that installs
    # them, which `tools/hooks/install-local-hooks.sh` does and which a clone can also read.
    Path("tools") / "ci",                # another ci script, e.g. clean_room.sh calling its checks
    Path("tools") / "hooks",             # a hook body
    # `scripts/runpod` USED TO BE HERE AND HAD TO COME OUT, 2026-09-28. The reasoning was that a
    # provisioning bootstrap gates an unattended job on a rented box, which is automated and
    # gating and so within what `ci` means. The reasoning was fine and the entry was not, because
    # `scripts/` is in `.git/info/exclude`: that bootstrap exists on one machine and in no clone.
    # So the guard read evidence no runner has, `artefact_ok.py` looked invoked here and was an
    # orphan everywhere else, and CI found it. An invoker nobody can fetch cannot make the claim
    # "run by CI or a release" true, which is why every entry is now checked against the index.
)


def _tracked_files():
    """Every path git is tracking, as repo-relative posix strings, or None if git cannot answer.

    THE EVIDENCE HAS TO BE EVIDENCE A CLONE HAS. This guard reads files to decide whether a script
    is invoked, and an untracked file is a fact about one machine. `scripts/runpod` and `.githooks`
    are both excluded from this repository, so both were contributing text that no runner, no
    container and no reviewer could ever see, and the check reported clean on that basis.

    That is the same defect the test below was written to catch, one level up in the machinery: the
    `__pycache__` case was a file vouching for itself, and this is a file vouching for a script in a
    tree nobody else has.
    """
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"],
                         capture_output=True, text=True, check=False, timeout=60)
    if out.returncode != 0:
        return None
    return {p for p in out.stdout.split("\0") if p}


def _text_of(rel, *, excluding=None, tracked=None):
    """Every file under `rel`, or the file itself, as one blob. Missing paths contribute nothing.

    `tracked`, when given, is the set of paths git knows about, and a file outside it contributes
    nothing either. See `_tracked_files` for why that is the difference between evidence and a fact
    about one machine.

    `excluding` drops one file from the blob, and it is load-bearing rather than tidy. `tools/ci` is
    itself an invoker, because one ci script legitimately calls another (`clean_room.sh` runs
    `clean_room_checks.py`). Without this, a script sitting in `tools/ci` VOUCHES FOR ITSELF: its own
    usage lines mention its own filename, the blob includes its own text, and the check passes.

    That is not hypothetical. The first version of this guard was written to catch a specific
    misfiled script, was mutation-tested by putting that script back, and PASSED. A guard that
    cannot fail on the case it was written for is worse than no guard, because the green tick is now
    evidence of something untrue.
    """
    def _readable(p):
        if p == excluding:
            return False
        return tracked is None or p.relative_to(ROOT).as_posix() in tracked

    target = ROOT / rel
    if target.is_file():
        return target.read_text(encoding="utf-8", errors="replace") if _readable(target) else ""
    if not target.is_dir():
        return ""
    # `__pycache__` IS EXCLUDED AND THAT IS NOT HOUSEKEEPING. A stale
    # `tools/ci/__pycache__/leak_sweep.cpython-314.pyc`, left behind from when a test imported the
    # script from that directory, contains the module name in its bytecode. So the compiled remains
    # of the very file being judged vouched for it, and this guard passed its own mutation test
    # twice for two different reasons before that was found. The candidate enumeration below always
    # skipped `__pycache__`; the evidence enumeration did not, which is the two-spellings problem
    # inside one test.
    return "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for p in sorted(target.rglob("*"))
                     if p.is_file() and _readable(p)
                     and "__pycache__" not in p.parts
                     and p.suffix not in (".pyc", ".pyo", ".so"))


def test_every_place_a_ci_script_may_be_invoked_from_is_one_a_clone_has():
    """The list below is only as good as the weakest entry, and one entry was local-only.

    WHAT PROMPTED IT, 2026-09-28

    `scripts/runpod` was added to `_INVOKERS` on the argument that an unattended provisioning
    bootstrap is a gate, which it is. What nobody checked is that `scripts/` sits in
    `.git/info/exclude`, so that bootstrap is on one machine and in no clone. `artefact_ok.py` was
    therefore invoked on this box and nowhere else, the orphan check read the local file and passed,
    and CI failed on the same commit the suite had called green.

    An untracked path cannot make "run by CI or a release" true for anybody but the person holding
    it. So the list is checked against the index, and a local-only entry fails HERE, naming the
    entry, rather than surfacing as a confusing orphan report about some other file.
    """
    tracked = _tracked_files()
    if tracked is None:
        pytest.skip("git cannot say what is tracked here, so this property is unobservable")

    local_only = [str(rel) for rel in _INVOKERS
                  if not any(p == str(rel).replace("\\", "/")
                             or p.startswith(str(rel).replace("\\", "/") + "/")
                             for p in tracked)]
    assert not local_only, (
        f"these are listed as places a tools/ci script may be invoked from, and git tracks nothing "
        f"under them: {local_only}.\n"
        f"  A file no clone has cannot make the `ci` claim true, and reading one makes this suite "
        f"green on a machine and red on every runner.\n"
        f"  Either the path is wrong, or the invocation belongs somewhere a clone can see.")


def test_every_ci_script_is_actually_run_by_ci_or_a_release():
    """`ci` means "checks an artefact or a tree, AND is run by CI or a release". Both halves.

    WHAT PROMPTED IT, 2026-09-27

    The `GROUPS` table defines each directory, and this file checked several properties of the
    scripts in them: that none sits loose at the top level, that the groups on disk match the
    documented ones, that root resolution is right in both Python and shell. It never checked the
    defining property of `ci` itself, which is the clause about being run.

    `tools/ci/leak_sweep.py` was the violation. It measures how much of an ablated direction the
    norm restore puts back: a model diagnostic, belonging in `research`, invoked by no workflow and
    named nowhere in `RELEASING.md`.

    It mattered more than filing usually does, because in this project "the leak gate" means the one
    control between a harmful prompt and a public push. Somebody auditing whether that gate runs in
    CI finds a `tools/ci/leak_sweep.py` that no workflow invokes, and draws a conclusion. Both
    available conclusions are wrong.

    A guard that checks three properties of a thing and not its definition is the same shape as a
    guard that covers one spelling of a defect: it reports clean, confidently, on the case that
    matters.
    """
    ci_dir = TOOLS / "ci"
    if not ci_dir.is_dir():
        pytest.skip("there is no tools/ci in this checkout")

    tracked = _tracked_files()
    if tracked is None:
        pytest.skip("git cannot say what is tracked here, and untracked evidence is not evidence")

    assert any(_text_of(rel, tracked=tracked).strip() for rel in _INVOKERS), (
        "none of the places that could invoke a ci script could be read, so this test would pass "
        f"whatever is in tools/ci. Looked for: {[str(r) for r in _INVOKERS]}")

    # Matched on the bare filename rather than the path, because workflows invoke these several
    # ways: `python tools/ci/x.py`, `./tools/ci/x.sh`, and from inside another ci script with a
    # relative path. The filenames here are distinctive enough that a bare match is safe.
    orphans = []
    for script in sorted(p for p in ci_dir.rglob("*")
                         if p.is_file() and p.suffix in (".py", ".sh")
                         and "__pycache__" not in p.parts):
        # Rebuilt per script, with that script's own text excluded, so it cannot cite itself.
        invokers = "\n".join(_text_of(rel, excluding=script, tracked=tracked)
                             for rel in _INVOKERS)
        if script.name not in invokers:
            orphans.append(script.relative_to(ROOT).as_posix())

    assert not orphans, (
        f"these live in tools/ci and nothing in CI, RELEASING.md or a hook invokes them: "
        f"{orphans}.\n"
        f'  tools/ci means "checks an artefact or a tree, AND is run by CI or a release". A '
        f"script nothing runs is not a check, it is a file that looks like one.\n"
        f"  Either wire it in, or move it: `research` for a diagnostic or experiment, `dev` for a "
        f"convenience run by hand.")
