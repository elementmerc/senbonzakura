"""The package's import surface: what `import senbonzakura` is allowed to drag in.

Every `python -m senbonzakura.<module>` run used to print

    RuntimeWarning: 'senbonzakura.track' found in sys.modules after import of
    package 'senbonzakura', but prior to execution of 'senbonzakura.track';
    this may result in unpredictable behaviour

because `__init__` imported `cli` and `cli` imports four sibling modules at
module level. runpy then executed a SECOND copy of the named module as
`__main__`. These tests pin the fix from both ends: the package stays cheap to
import, and the two names it re-exports keep working.

The subprocess is load bearing. A test in the same interpreter as the rest of
the suite proves nothing here, because by then something else has already
imported `cli` and `sys.modules` is populated either way.
"""

import subprocess
import sys

import pytest

# Every module a run spec invokes with `-m`.
ENTRY_MODULES = ["track", "score", "margin", "coherence", "cli"]


def _run(code):
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_importing_the_package_does_not_pull_in_the_cli():
    r = _run("import senbonzakura, sys; print('senbonzakura.cli' in sys.modules)")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False", (
        "importing the package imported cli, which re-imports four siblings and "
        "puts every `-m` entry point back in the double-import warning"
    )


@pytest.mark.parametrize("module", ENTRY_MODULES)
def test_importing_the_package_does_not_pull_in_an_entry_module(module):
    r = _run(f"import senbonzakura, sys; print('senbonzakura.{module}' in sys.modules)")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"


def test_the_package_still_re_exports_main_and_version():
    r = _run(
        "import senbonzakura;"
        " assert callable(senbonzakura.main), 'main is not callable';"
        " assert isinstance(senbonzakura.__version__, str), 'version is not a string';"
        " print(senbonzakura.__version__)"
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip()


def test_an_unknown_attribute_still_raises_attribute_error():
    r = _run(
        "import senbonzakura\n"
        "try:\n"
        "    senbonzakura.no_such_name\n"
        "except AttributeError as e:\n"
        "    print('AttributeError:', e)\n"
        "else:\n"
        "    raise SystemExit('a missing attribute did not raise')\n"
    )
    assert r.returncode == 0, r.stderr
    assert "no_such_name" in r.stdout


def test_dir_lists_the_lazy_re_exports():
    """A lazy name that `dir()` cannot see is invisible to tab-completion."""
    import senbonzakura

    names = dir(senbonzakura)
    assert "main" in names
    assert "__version__" in names


@pytest.mark.parametrize("module", ENTRY_MODULES)
def test_the_entry_modules_run_without_a_double_import_warning(module):
    """The end-to-end check, run exactly as a spec runs it."""
    r = subprocess.run(
        [sys.executable, "-m", f"senbonzakura.{module}", "--help"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "RuntimeWarning" not in r.stderr, r.stderr
    assert "found in sys.modules" not in r.stderr, r.stderr


# ── the licence notice has to be in the artefact, not only in the repository ───────────
def test_the_statement_of_modification_ships_inside_the_distribution():
    """AGPL section 5(a) asks the DISTRIBUTED work to say it is modified, not the git tree.

    Senbonzakura includes code copied from Heretic, so the whole project is AGPL and section 5(a)
    requires a prominent notice of modification. That notice lives in THIRD-PARTY-NOTICES.md. The
    0.3.0 wheel on PyPI listed LICENSE alone, so anyone who installed it received the licence and
    no statement that this is a modified work: the statement existed exactly where it was not
    needed. This asserts the file is listed for packaging AND still contains the section it is
    listed for.
    """
    from pathlib import Path

    import tomllib

    root = Path(__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as f:
        cfg = tomllib.load(f)

    listed = cfg["project"]["license-files"]
    assert "THIRD-PARTY-NOTICES.md" in listed, (
        "the statement of modification is not packaged, so an installed copy carries none")

    for name in listed:
        assert (root / name).is_file(), f"{name} is listed for packaging but does not exist"

    notice = (root / "THIRD-PARTY-NOTICES.md").read_text(encoding="utf-8")
    assert "5(a)" in notice, "the notice no longer carries the statement it is packaged for"
    assert "Heretic" in notice, "the notice no longer names the work this one is derived from"


def test_the_modification_date_is_not_older_than_the_code_it_describes():
    """AGPL section 5(a) asks for a notice of modification AND a relevant date.

    A date is only a date if it is true. The notice recorded 2026-07-29 while the file carrying
    it had been modified on 2026-08-05, so the shipped artefact told a reader the work had not
    been touched for a week when it had. Nobody would have noticed, because nothing checked: the
    date was maintained by remembering, which is the same mechanism that let the notice itself
    ship outside the wheel for a whole release.

    The assertion is one-sided on purpose. The recorded date may run ahead of the last commit,
    since the commit that updates the notice is itself the modification being described, and a
    working tree can legitimately be ahead of git. What it may never do is fall behind.
    """
    import datetime
    import re
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    carrier = root / "src" / "senbonzakura" / "metrics.py"

    text = carrier.read_text(encoding="utf-8")
    m = re.search(r"last modified (\d{4}-\d{2}-\d{2})", text)
    assert m, "the section 5(a) notice in metrics.py no longer records a modification date"
    recorded = datetime.date.fromisoformat(m.group(1))

    r = subprocess.run(
        ["git", "log", "-1", "--format=%cd", "--date=short", "--", str(carrier)],
        capture_output=True, text=True, cwd=root, check=False, timeout=60)
    if r.returncode != 0 or not r.stdout.strip():
        pytest.skip("no git history here, so the recorded date cannot be checked against it")
    committed = datetime.date.fromisoformat(r.stdout.strip())

    assert recorded >= committed, (
        f"the notice says the file was last modified {recorded}, but it was committed on "
        f"{committed}. Section 5(a) asks for a relevant date and this one is stale.")


def test_the_two_places_the_date_is_written_agree():
    """It appears in the source notice and in the packaged statement, and both are distributed."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    src = re.search(r"last modified (\d{4}-\d{2}-\d{2})",
                    (root / "src" / "senbonzakura" / "metrics.py").read_text(encoding="utf-8"))
    notices = re.search(r"Most recent modification to the file carrying it:\*\* (\d{4}-\d{2}-\d{2})",
                        (root / "THIRD-PARTY-NOTICES.md").read_text(encoding="utf-8"))
    assert src and notices, "one of the two statements of modification no longer carries a date"
    assert src.group(1) == notices.group(1), (
        f"metrics.py says {src.group(1)} and THIRD-PARTY-NOTICES.md says {notices.group(1)}; "
        f"an installed copy would carry two different answers to the same question")


def test_every_module_in_the_source_tree_reaches_the_distribution():
    """The compass shipped in no released artefact, and nothing would have said so.

    `pip install senbonzakura` at 0.3.0 gets a package with no `margin.py`, so the measurement
    this project is about cannot be run from the thing anyone installs. That particular case is
    an accident of timing rather than of packaging: the tag predates the compass, and the fix is
    the next release rather than a code change. Verified 2026-08-06 by building a wheel from this
    tree and reading it, which carried all twelve modules and both licence files.

    This asserts the property that fix depends on: nothing in the package directory is excluded
    from packaging. setuptools includes a found package wholesale today, so the guard is against
    a future `exclude` rule, a `MANIFEST.in`, or a move to a build backend that does not, any of
    which would drop a module silently and produce exactly the same symptom again.
    """
    from pathlib import Path

    import tomllib

    root = Path(__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as f:
        cfg = tomllib.load(f)

    find = cfg["tool"]["setuptools"].get("packages", {}).get("find", {})
    assert not find.get("exclude"), (
        f"packaging excludes {find['exclude']}, so a module in the source tree may not reach an "
        f"installed copy. Every module here is part of the tool; none is a test fixture.")
    assert not (root / "MANIFEST.in").exists(), (
        "a MANIFEST.in has appeared; it can narrow what ships, and this project has already "
        "released a wheel missing the module its central claim depends on")

    on_disk = {p.stem for p in (root / "src" / "senbonzakura").glob("*.py")}
    for required in ("margin", "crashsafe", "track", "validate", "metrics", "cli"):
        assert required in on_disk, f"{required}.py has left the package directory"


def test_the_readme_only_documents_entry_points_that_exist():
    """A documented command that imports nothing is a bug report waiting to be filed."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    documented = set(re.findall(r"python -m senbonzakura\.([a-z_]+)",
                                (root / "README.md").read_text(encoding="utf-8")))
    assert documented, "the README documents no module entry points at all any more"
    for module in documented:
        assert (root / "src" / "senbonzakura" / f"{module}.py").is_file(), (
            f"the README tells a reader to run `python -m senbonzakura.{module}`, "
            f"and no such module exists")


def test_no_committed_file_carries_a_home_directory_path():
    """A public repository must not carry a home directory path, which names an account.

    Found 2026-08-06 in two committed compass results, which recorded the absolute path of the
    dataset they measured. The path was genuinely useful (a reader wants to know WHICH set was
    scored) and the leading half of it was not, so the fix keeps the last two components and drops
    the rest rather than deleting the field.

    This is a gate rather than a one-off scrub, because the tool writes these files and will write
    more of them. It reads the git index rather than the working tree, so an artefact staged with
    a path and tidied on disk afterwards is still caught, which is the same reasoning as the
    prompt-retention hook.
    """
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    r = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=root,
                       check=False, timeout=120)
    if r.returncode != 0:
        pytest.skip("not a git checkout, so there is no index to read")

    offenders = []
    for name in r.stdout.split("\n"):
        if not name.strip():
            continue
        f = root / name
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        # Built rather than written out, so this file does not trip its own check. The first
        # version did, which was a good sign about the check and a bad one about the author.
        needle = "/" + "home" + "/"
        allowed = needle + "runner"   # GitHub Actions' own directory: a container, not a person
        for line_no, line in enumerate(text.splitlines(), 1):
            if needle in line and allowed not in line:
                offenders.append(f"{name}:{line_no}")
    assert not offenders, (
        "committed files carry a home directory path, which names an account on somebody's "
        "machine in a public repository: " + ", ".join(offenders[:8]))
