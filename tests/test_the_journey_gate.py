# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Tests for the journey gate: tools/ci/check_journeys.py and scripts/hooks/journey-check.sh.

The suite this gate guards exists because three defects escaped 6,500 unit tests on 2026-09-28,
each of them living in a SEQUENCE rather than in a function (decision Q-43). So the gate's own
failure modes are the ones worth testing, and they are all the same shape: a gate that reports a
number which is not true.

- a journey with no mark counts as a test while covering nothing, so the measured coverage falls
  while the file looks like work;
- a misspelt surface counts as covered while covering nothing at all;
- a hand-typed command inventory rots, and a rotted inventory reports a percentage that is TOO
  HIGH, which is worse than no percentage;
- a floor that can be wound back quietly is not a floor.

Every test below builds its own journey tree under tmp_path rather than reading the real
tests/journeys/, so none of them changes meaning when a journey is added tomorrow.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "check_journeys", REPO / "tools" / "ci" / "check_journeys.py")
cj = importlib.util.module_from_spec(_SPEC)
sys.modules["check_journeys"] = cj
_SPEC.loader.exec_module(cj)

HOOK = REPO / "scripts" / "hooks" / "journey-check.sh"

#: SKIPPED WHERE THE HOOK IS NOT, and that is the normal case rather than a broken one.
#:
#: `scripts/` is excluded from git in this repository, per baseline Section 26: it has a public
#: remote, and naming a private tree in a committed `.gitignore` tells every visitor it exists. So
#: the hook is per clone, and a fresh checkout, which is every CI runner, does not carry it. These
#: tests failed on every CI job with `bash: ... No such file or directory` before this existed.
#:
#: A skip rather than a deletion, because the hook IS tested where it lives, which is the machine
#: somebody is working on. The reason is printed by `-rs`, so a reader sees that this was not run
#: rather than reading it as a pass.
needs_the_hook = pytest.mark.skipif(
    not HOOK.is_file(),
    reason=("scripts/hooks/journey-check.sh is not in this checkout. It is excluded from git "
            "(baseline Section 26, public remote), so it is per clone and CI never has it. The "
            "hook is exercised on the machine that carries it."))


@pytest.fixture
def journeys(tmp_path):
    """Write one journey module into a fresh tests/journeys tree and return the directory."""
    directory = tmp_path / "journeys"
    directory.mkdir()

    def write(body, name="test_walk.py"):
        (directory / name).write_text("import pytest\n\n\n" + body, encoding="utf-8")
        return directory
    return write


@pytest.fixture
def fake_src(tmp_path):
    """A minimal src/ tree carrying the three tables the command universe is derived from."""
    def write(delegated=("doctor",), aliases=(), modes=("abliterate",)):
        pkg = tmp_path / "src" / "senbonzakura"
        pkg.mkdir(parents=True, exist_ok=True)
        rows = "".join(f'    "{c}": ("m", "main"),\n' for c in delegated)
        alias_rows = "".join(f'    "{a}": "doctor",\n' for a in aliases)
        (pkg / "entry.py").write_text(
            f"DELEGATED: dict[str, tuple[str, str]] = {{\n{rows}}}\n"
            f"ALIASES = {{\n{alias_rows}}}\n"
            f'RETIRED = {{"bench": "head-to-head"}}\n', encoding="utf-8")
        (pkg / "parser.py").write_text(
            "MODES = (" + "".join(f'"{m}", ' for m in modes) + ")\n", encoding="utf-8")
        return str(tmp_path / "src")
    return write


# ── a journey that declares nothing ───────────────────────────────────────────────────────────

def test_a_journey_with_no_mark_is_refused_by_name(journeys):
    """An undeclared journey lowers the measured coverage while looking on the page like work.

    It is the quietest way for the suite to stop meaning anything: the count of test functions
    keeps rising and the number of surfaces they are credited with does not.
    """
    directory = journeys("def test_a_walk_nobody_declared(session):\n    pass\n")
    with pytest.raises(cj.JourneyError) as e:
        cj.read_journeys(directory)
    assert "test_a_walk_nobody_declared" in str(e.value)
    assert "no `journey` mark" in str(e.value)


def test_a_mark_that_names_no_surfaces_is_refused(journeys):
    """`@pytest.mark.journey` with empty parentheses declares nothing and must not read as a
    declaration. Same defect as no mark at all, wearing the mark's clothes.
    """
    directory = journeys("@pytest.mark.journey()\ndef test_empty(session):\n    pass\n")
    with pytest.raises(cj.JourneyError, match="names no surfaces"):
        cj.read_journeys(directory)


def test_a_bare_mark_with_no_call_is_refused(journeys):
    """`@pytest.mark.journey` without parentheses is legal pytest and declares nothing."""
    directory = journeys("@pytest.mark.journey\ndef test_bare(session):\n    pass\n")
    with pytest.raises(cj.JourneyError, match="names no surfaces"):
        cj.read_journeys(directory)


def test_a_helper_that_is_not_a_test_is_not_a_journey(journeys):
    """Only `test_*` functions are journeys. A module-level helper needing a mark would make the
    gate refuse every journey file that factors anything out.
    """
    directory = journeys(
        "def build_a_session():\n    pass\n\n\n"
        '@pytest.mark.journey("command:doctor")\n'
        "def test_one(session):\n    pass\n")
    assert len(cj.read_journeys(directory)) == 1


# ── a surface string that is not a surface ────────────────────────────────────────────────────

def test_a_malformed_surface_names_the_file_and_the_function(journeys):
    """A surface written in the wrong vocabulary covers nothing while counting as something, so
    the refusal has to say which function to open.
    """
    directory = journeys(
        '@pytest.mark.journey("doctor")\ndef test_the_doctor(session):\n    pass\n',
        name="test_front.py")
    with pytest.raises(cj.JourneyError) as e:
        cj.read_journeys(directory)
    message = str(e.value)
    assert "test_front.py" in message
    assert "test_the_doctor" in message
    assert "'doctor'" in message


def test_a_surface_built_at_runtime_is_refused(journeys):
    """The gate reads source rather than importing it, so a computed surface is unreadable.

    It refuses rather than skipping, because a skipped mark is an undeclared journey.
    """
    directory = journeys(
        "NAME = 'command:doctor'\n\n\n"
        "@pytest.mark.journey(NAME)\ndef test_computed(session):\n    pass\n")
    with pytest.raises(cj.JourneyError, match="cannot read"):
        cj.read_journeys(directory)


def test_a_well_formed_surface_that_is_in_no_universe_is_refused(journeys, fake_src, tmp_path,
                                                                 capsys):
    """`state:no_tty` for `state:no-tty` is shaped correctly and covers nothing. Left unchecked it
    inflates the covered count and the ratchet locks the inflation in.
    """
    directory = journeys(
        '@pytest.mark.journey("state:no_tty")\ndef test_a_typo(session):\n    pass\n')
    ratchet = tmp_path / "ratchet.json"
    ratchet.write_text('{"journeys": 0, "surfaces": 0}', encoding="utf-8")
    status = cj.main(["--journeys", str(directory), "--src", fake_src(),
                      "--ratchet", str(ratchet)])
    assert status == 1
    assert "state:no_tty" in capsys.readouterr().err


# ── the universe is derived, never typed ──────────────────────────────────────────────────────

def test_a_command_added_to_the_table_appears_in_the_universe_untouched(fake_src):
    """Adding a command to `entry.DELEGATED` must put it in the universe without anybody editing
    the gate. A hand-typed inventory rots, and a rotted one reports a percentage that is too high.
    """
    surfaces = cj.universe(fake_src(delegated=("doctor", "nobody-has-written-this-yet")))
    assert "command:nobody-has-written-this-yet" in surfaces


def test_aliases_and_modes_are_surfaces_and_retired_words_are_not(fake_src):
    """An alias and a mode are words a person types, so each is walkable. A retired word is kept
    only so the tool can say what it was renamed to; journeying it would be covering a ghost.
    """
    surfaces = cj.universe(fake_src(delegated=("doctor",), aliases=("harm-recognition",),
                                    modes=("abliterate", "auto")))
    assert {"command:harm-recognition", "command:abliterate", "command:auto"} <= surfaces
    assert "command:bench" not in surfaces


def test_the_bare_invocation_and_the_short_flag_are_surfaces(fake_src):
    """Neither appears in any command table and both are the first thing a newcomer types."""
    surfaces = cj.universe(fake_src())
    assert {"command:senbonzakura", "command:-i"} <= surfaces


def test_the_universe_reads_the_real_tables_of_this_repository():
    """The fixture tests above would all pass against a table this gate could no longer find, so
    one test holds it against the shipped source.
    """
    surfaces = cj.universe(REPO / "src")
    assert {"command:doctor", "command:head-to-head", "command:kageyoshi"} <= surfaces
    assert "state:no-tty" in surfaces


def test_a_missing_command_table_is_a_loud_refusal_not_a_smaller_universe(tmp_path):
    """If `DELEGATED` is renamed, the honest answer is that the gate has stopped measuring. A
    silently smaller universe would raise the coverage percentage on a change that covered
    nothing.
    """
    pkg = tmp_path / "src" / "senbonzakura"
    pkg.mkdir(parents=True)
    (pkg / "entry.py").write_text("SOMETHING_ELSE = {}\n", encoding="utf-8")
    (pkg / "parser.py").write_text('MODES = ("abliterate",)\n', encoding="utf-8")
    with pytest.raises(cj.JourneyError, match="DELEGATED"):
        cj.universe(tmp_path / "src")


# ── the ratchet ───────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def gate(tmp_path, journeys, fake_src):
    """Run main() over a tree with `count` journeys and a ratchet at the given high-water marks."""
    ratchet = tmp_path / "ratchet.json"

    def run(count, floor=None, extra=()):
        body = "".join(
            f'@pytest.mark.journey("command:doctor")\ndef test_walk_{i}(session):\n    pass\n\n\n'
            for i in range(count))
        directory = journeys(body or "# no journeys yet\n")
        if floor is not None:
            ratchet.write_text(json.dumps({"journeys": floor[0], "surfaces": floor[1]}),
                               encoding="utf-8")
        return cj.main(["--journeys", str(directory), "--src", fake_src(),
                        "--ratchet", str(ratchet), *extra]), ratchet
    return run


def test_the_gate_holds_when_nothing_has_fallen(gate, capsys):
    status, _ = gate(3, floor=(3, 1))
    assert status == 0
    assert "floor held" in capsys.readouterr().out


def test_the_gate_refuses_when_the_journey_count_falls(gate, capsys):
    """Deleting a journey is the cheapest way to make a red suite green, so it is the one the
    floor exists to catch, and the refusal says by how much.
    """
    status, _ = gate(2, floor=(5, 1))
    assert status == 1
    err = capsys.readouterr().err
    assert "journeys fell from 5 to 2" in err
    assert "down 3" in err


def test_the_gate_refuses_when_the_covered_surface_count_falls(gate, capsys):
    """A suite can keep every test and still stop covering a surface, by rewriting one journey's
    marks. The count of journeys would not move; this number would.
    """
    status, _ = gate(5, floor=(5, 4))
    assert status == 1
    assert "surfaces fell from 4 to 1" in capsys.readouterr().err


def test_update_refuses_to_lower_a_number_without_force(gate, capsys):
    """A ratchet that can be quietly wound back is not a ratchet: it is a record of the last time
    somebody ran the tool.
    """
    status, ratchet = gate(2, floor=(9, 3), extra=["--update"])
    assert status == 1
    assert "would go from 9 to 2" in capsys.readouterr().err
    assert json.loads(ratchet.read_text(encoding="utf-8"))["journeys"] == 9


def test_update_with_force_lowers_it(gate):
    status, ratchet = gate(2, floor=(9, 3), extra=["--update", "--force", "--today", "2026-09-28"])
    assert status == 0
    recorded = json.loads(ratchet.read_text(encoding="utf-8"))
    assert recorded["journeys"] == 2
    assert recorded["updated"] == "2026-09-28"


def test_update_raises_a_number_without_force(gate):
    status, ratchet = gate(6, floor=(4, 1), extra=["--update"])
    assert status == 0
    assert json.loads(ratchet.read_text(encoding="utf-8"))["journeys"] == 6


def test_a_ratchet_that_cannot_be_read_is_a_refusal_not_a_zero_floor(gate, tmp_path, capsys):
    """Treating an unreadable floor as zero would turn a corrupt file into a permanently passing
    gate, which is the failure that looks most like success.
    """
    status, ratchet = gate(1, floor=(1, 1))
    assert status == 0
    ratchet.write_text("{not json", encoding="utf-8")
    assert cj.main(["--journeys", str(tmp_path / "journeys"), "--src",
                    str(tmp_path / "src"), "--ratchet", str(ratchet)]) == 1
    assert "cannot be read" in capsys.readouterr().err


def test_a_missing_ratchet_file_refuses_and_says_what_to_run(tmp_path, capsys):
    status = cj.main(["--journeys", str(tmp_path), "--src", str(REPO / "src"),
                      "--ratchet", str(tmp_path / "nope.json")])
    assert status == 1
    assert "--update" in capsys.readouterr().err


def test_the_report_names_the_uncovered_surfaces(gate, capsys):
    """A percentage on its own is decoration. The next journey is chosen off this list."""
    gate(1, floor=(0, 0))
    out = capsys.readouterr().out
    assert "surfaces: 1 of " in out
    assert "state:resumed" in out.split("uncovered:", 1)[1]


def test_the_shipped_ratchet_file_is_readable():
    """The committed floor is what CI reads; a typo in it turns the gate off."""
    counts = cj.read_ratchet(REPO / "journeys" / "ratchet.json")
    assert counts["journeys"] >= 0 and counts["surfaces"] >= 0


# ── the advisory hook: it fires on changed surface, and only on changed surface ────────────────

@pytest.fixture
def repo(tmp_path):
    """A one-commit git repository, and a helper that changes files and runs the hook over them."""
    work = tmp_path / "work"
    (work / "src" / "senbonzakura").mkdir(parents=True)
    (work / "tests" / "journeys").mkdir(parents=True)

    def git(*args):
        return subprocess.run(["git", *args], cwd=work, check=True, capture_output=True, text=True)

    git("init", "-q", "-b", "dev")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "T")
    for path in ("src/senbonzakura/parser.py", "src/senbonzakura/abliterator.py",
                 "tests/journeys/test_walk.py"):
        (work / path).write_text("# base\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "base")
    # A second commit, so that HEAD^ resolves and the hook has a range to read. With one commit
    # it exits early having measured nothing, and every test below would pass on a hook that does
    # nothing at all.
    (work / "README.md").write_text("# nothing a person sees\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "second")

    def run(*changed, commit=True):
        for path in changed:
            (work / path).write_text("# changed\n", encoding="utf-8")
        if commit:
            git("add", "-A")
            git("commit", "-qm", "change")
        return subprocess.run(["bash", str(HOOK)], cwd=work, capture_output=True, text=True,
                              env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
                                   "CLAUDE_PROJECT_DIR": str(work)}, check=False)
    return run


@needs_the_hook
def test_the_hook_asks_for_a_journey_when_a_user_facing_file_moved(repo):
    """The prompt fires on CHANGED SURFACE, which is what Q-43 chose over a commit count."""
    done = repo("src/senbonzakura/parser.py")
    assert done.returncode == 0
    assert "parser.py" in done.stdout
    assert "tests/journeys/" in done.stdout


@needs_the_hook
def test_the_hook_says_nothing_when_only_internals_moved(repo):
    """Credibility is the only property a gate has. One that speaks when nothing a person can see
    has changed is one people learn to scroll past, and then it is not there on the day it
    matters.
    """
    done = repo("src/senbonzakura/abliterator.py")
    assert done.returncode == 0
    assert done.stdout.strip() == ""


@needs_the_hook
def test_the_hook_says_nothing_when_a_journey_moved_with_the_surface(repo):
    """The prompt has already been answered, so asking again is noise."""
    done = repo("src/senbonzakura/parser.py", "tests/journeys/test_walk.py")
    assert done.returncode == 0
    assert done.stdout.strip() == ""


@needs_the_hook
def test_a_brand_new_untracked_journey_answers_the_prompt(repo):
    """The first journey anybody writes is a file git diff cannot see, because it is untracked.

    Without counting untracked files the hook asks for a journey at the exact moment somebody has
    one open unsaved to the index, which is the fastest way to teach a person to ignore it.
    """
    done = repo("src/senbonzakura/parser.py", "tests/journeys/test_brand_new.py", commit=False)
    assert done.returncode == 0
    assert done.stdout.strip() == ""


@needs_the_hook
def test_the_hook_is_silent_and_successful_outside_a_repository(tmp_path):
    """Fail-open: an advisory hook must never be the reason a session or a commit stops."""
    done = subprocess.run(["bash", str(HOOK)], cwd=tmp_path, capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
                               "CLAUDE_PROJECT_DIR": str(tmp_path)}, check=False)
    assert done.returncode == 0
    assert done.stdout.strip() == ""


@needs_the_hook
def test_the_surface_list_lives_in_exactly_one_place():
    """Two copies drift, and the copy that drifts is always the one deciding whether it fires."""
    text = HOOK.read_text(encoding="utf-8")
    assert text.count("src/senbonzakura/interactive.py") == 1
