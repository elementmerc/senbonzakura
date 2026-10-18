# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
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
def test_the_third_party_notices_ship_inside_the_distribution():
    """The notices have to reach the DISTRIBUTED work, not only the git tree.

    The 0.3.0 wheel on PyPI listed LICENSE alone, so anyone who installed it received the licence
    and none of the notices. This asserts the file is listed for packaging and still carries the
    credits it is listed for. (It carried an AGPL section 5(a) statement of modification while
    Heretic's code was included; that code was removed on 2026-10-18.)
    """
    from pathlib import Path

    from tomlread import tomllib

    root = Path(__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as f:
        cfg = tomllib.load(f)

    listed = cfg["project"]["license-files"]
    assert "THIRD-PARTY-NOTICES.md" in listed, (
        "the statement of modification is not packaged, so an installed copy carries none")

    for name in listed:
        assert (root / name).is_file(), f"{name} is listed for packaging but does not exist"

    notice = (root / "THIRD-PARTY-NOTICES.md").read_text(encoding="utf-8")
    assert "Heretic" in notice, "the notice no longer credits the work the search builds on"


def test_no_code_copied_from_heretic_ships_in_the_package():
    """Heretic's keyword marker list and its normalisation were removed on 2026-10-18 (the operator's
    decision), because they were the one piece of code in this package written by someone else
    under AGPL. This keeps them from coming back by accident: the names are gone from the module,
    and the notices no longer say any of Heretic's code is included.
    """
    from pathlib import Path

    from senbonzakura import metrics
    for name in ("HERETIC_MARKERS", "_heretic_norm", "is_heretic_refusal", "heretic_keyword_rate",
                 "KNEE_W_KEYWORD"):
        assert not hasattr(metrics, name), f"metrics.{name} is back"
    root = Path(__file__).resolve().parent.parent
    notices = " ".join((root / "THIRD-PARTY-NOTICES.md").read_text(encoding="utf-8").split())
    assert "copied verbatim from Heretic" not in notices, (
        "THIRD-PARTY-NOTICES.md says Heretic code is included, and none is")


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

    from tomlread import tomllib

    root = Path(__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as f:
        cfg = tomllib.load(f)

    find = cfg["tool"]["setuptools"].get("packages", {}).get("find", {})
    assert not find.get("exclude"), (
        f"packaging excludes {find['exclude']}, so a module in the source tree may not reach an "
        f"installed copy. Every module here is part of the tool; none is a test fixture.")
    # A MANIFEST.in is allowed, and what it is allowed to remove is not. It exists to prune
    # `vendor/bin` from the SOURCE distribution, which carried 33 Linux binaries inside an
    # artefact that has no platform tag by construction. Anything that could drop a module is
    # still the failure this test was written for, so the ban is on the rules rather than on the
    # file: no exclusion may name a Python file, and none may reach into the package beyond the
    # vendored binaries.
    manifest = root / "MANIFEST.in"
    if manifest.exists():
        for raw in manifest.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            verb, _, rest = line.partition(" ")
            if verb not in ("prune", "exclude", "recursive-exclude", "global-exclude"):
                continue
            assert ".py" not in rest, (
                f"MANIFEST.in line {line!r} excludes Python files, which is how a wheel comes to "
                f"be missing the module its central claim depends on")
            assert rest.strip().startswith("src/senbonzakura/vendor/bin"), (
                f"MANIFEST.in line {line!r} removes something outside the vendored binaries. "
                f"Only those are meant to be pruned; widening it silently drops shipped code.")

    on_disk = {p.stem for p in (root / "src" / "senbonzakura").glob("*.py")}
    for required in ("margin", "crashsafe", "track", "validate", "metrics", "cli"):
        assert required in on_disk, f"{required}.py has left the package directory"


def test_the_readme_only_documents_entry_points_that_exist():
    """A documented command that does not exist is a bug report waiting to be filed.

    Both spellings are checked, because the README uses the console script (`senbonzakura track`)
    and the docs sometimes use the module form (`python -m senbonzakura.track`). The check went
    briefly vacuous when the README was cut down to an entry point and switched spelling, which
    is the failure mode every guard in this file exists to make loud.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text(encoding="utf-8")

    modules = set(re.findall(r"python -m senbonzakura\.([a-z_]+)", readme))
    for module in modules:
        assert (root / "src" / "senbonzakura" / f"{module}.py").is_file(), (
            f"the README says `python -m senbonzakura.{module}` and no such module exists")

    # The console-script form, e.g. `senbonzakura track --harmful ...`.
    from senbonzakura import cli
    known = set(cli.DELEGATED) | {"abliterate", "kageyoshi"}
    commands = set(re.findall(r"(?:^|\s)senbonzakura ([a-z-]+)", readme)) - {"--help"}
    for command in commands:
        assert command in known, (
            f"the README documents `senbonzakura {command}`, which is not a command. "
            f"Known: {', '.join(sorted(known))}")

    assert modules or commands, (
        "the README documents no runnable command at all any more, so this check is passing "
        "over an empty set rather than checking anything")


# ── the exit code has to survive the way out ─────────────────────────────────────
#
# `__main__.py` called `main()` and dropped the return value, so `python -m senbonzakura`
# reported success for any command that fails by RETURNING a status rather than raising.
# `doctor` printed "this install cannot do what it claims" over nine failed checks and exited 0,
# which makes it useless as the pre-flight gate it was built to be. Only a subprocess can show
# this: calling `main()` in-process returns the number perfectly well, and the loss happens on
# the way out of the interpreter.
def _run_module(args, stdin=""):
    return subprocess.run(
        [sys.executable, "-m", "senbonzakura", *args],
        capture_output=True, text=True, timeout=300, check=False, input=stdin,
    )


def test_a_command_that_returns_a_failure_code_exits_with_it():
    """`interactive` refuses a non-terminal stdin by returning 2. Piped stdin is never a TTY,
    so this is deterministic without needing a pty.
    """
    r = _run_module(["interactive"])
    assert r.returncode == 2, (
        f"the refusal exited {r.returncode}; a caller checking the status saw success. "
        f"stdout={r.stdout[-300:]!r}"
    )


def test_python_dash_m_agrees_with_the_console_script():
    """The two documented invocations must not disagree about whether a run failed."""
    import shutil
    exe = shutil.which("senbonzakura")
    if exe is None:                                        # pragma: no cover
        pytest.skip("the console script is not on PATH in this environment")
    direct = subprocess.run([exe, "interactive"], capture_output=True, text=True,
                            timeout=300, check=False, input="")
    assert _run_module(["interactive"]).returncode == direct.returncode


def test_success_still_exits_zero():
    """`sys.exit(main())` must not turn a normal run into a failure: the abliterate paths
    return None, and None is exit 0.
    """
    assert _run_module(["--help"]).returncode == 0
