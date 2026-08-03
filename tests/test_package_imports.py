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
