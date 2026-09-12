# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The checker is a separate distribution, and the separation is a property rather than a plan.

DECISION Q-29 (2026-09-11). `roadmap.md` scores this project against the six properties shared
by tools that took a domain in under two years and calls "adoption cost near zero" the largest
single gap: installing senbonzakura means torch and a GPU, where every fast comparator installs
in seconds and touches nothing. `senbonzakura-check` is that distribution.

THE FAILURE THIS FILE EXISTS TO PREVENT

Nothing goes wrong on any developer machine when somebody adds `from senbonzakura import
something` to a module under `senbonzakura_check/`. The suite passes, the command works, and the
torch-free property is silently gone until a user on a laptop discovers that `pip install
senbonzakura-check` now wants a gigabyte. That is the same shape as every defect this project
has had to withdraw a number over: correct-looking, and only wrong somewhere nobody was looking.

So the direction of the dependency is asserted, the metadata is asserted, and the version
agreement between the two distributions is asserted, because all three are invisible.

WHAT IS NOT ASSERTED HERE, and is owed: resolving the checker's dependencies in a clean
environment and confirming torch is absent from the resolution. That needs a network and a
build, so it lives in CI rather than in the suite, and `DEFERRED.md` carries it until it does.
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest
from tomlread import tomllib

#: Imports the checker and prints every `senbonzakura*` module that ended up in `sys.modules`.
#: A separate constant so the string is not implicitly concatenated inside an argument list,
#: where a missing comma silently becomes a different command.
PROBE = (
    "import senbonzakura_check as c; "
    "assert c.load_checks(); "
    "print(sorted(m for m in __import__('sys').modules if m.startswith('senbonzakura')))"
)

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "checker"


def _pyproject(path):
    with path.open("rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="module")
def checker_cfg():
    return _pyproject(CHECKER / "pyproject.toml")


@pytest.fixture(scope="module")
def main_cfg():
    return _pyproject(ROOT / "pyproject.toml")


# ── the dependency list, which is the product ────────────────────────────────────────────────

def test_the_checker_declares_no_dependencies_at_all(checker_cfg):
    """NOT "small". EMPTY.

    Every fast comparator in the roadmap's table installs in seconds and touches nothing, and
    this is the distribution that has to do the same. Adding anything here is a decision rather
    than a convenience, and it costs the one property this distribution exists to hold.
    """
    assert checker_cfg["project"]["dependencies"] == [], (
        "the torch-free distribution has grown a dependency; if a check genuinely needs one, "
        "the check does not ship (v0.8 plan, loophole 1)")


def test_the_checker_declares_no_optional_dependencies_either(checker_cfg):
    """An extra is a dependency somebody will install, and `pip install senbonzakura-check[all]`
    quietly becoming the documented invocation would lose the property by the back door.
    """
    assert not checker_cfg["project"].get("optional-dependencies")


def test_the_big_distribution_depends_on_the_small_one(main_cfg):
    """Which is what makes `senbonzakura check` work from a full install.

    It is also the only direction allowed. Reversing it would mean the checker pulling the
    abliterator, which is the whole thing being avoided.
    """
    deps = main_cfg["project"]["dependencies"]
    assert any(d.split(">")[0].split("=")[0].strip() == "senbonzakura-check" for d in deps), (
        "the abliterator no longer depends on the checker, so `senbonzakura check` would be a "
        "command that imports something nobody installed")


# ── the direction of the imports, which is invisible and therefore gated ─────────────────────

def _imports_of(pkg):
    out = []
    for py in sorted(pkg.rglob("*.py")):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                out += [(py, node.lineno, a.name) for a in node.names]
            # A relative import cannot reach out of the package, so level 0 only.
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                out.append((py, node.lineno, node.module or ""))
    return out


def test_nothing_in_the_checker_imports_the_abliterator():
    """THE ONE RULE THE SPLIT DEPENDS ON, and it fails silently if left to discipline.

    An upward import works perfectly in this repository, where both packages are installed, and
    breaks only for somebody who installed `senbonzakura-check` alone: exactly the person the
    distribution exists for, and the one person who never appears in our CI.
    """
    pkg = CHECKER / "src" / "senbonzakura_check"
    offenders = [f"{p.relative_to(ROOT)}:{line} imports {name}"
                 for p, line, name in _imports_of(pkg)
                 if name == "senbonzakura" or name.startswith("senbonzakura.")]
    assert not offenders, (
        "the checker imports the abliterator, which puts torch back into the torch-free "
        "distribution for anyone who installed it alone:\n  " + "\n  ".join(offenders))


def test_nothing_in_the_checker_imports_anything_heavy():
    """The stdlib-only claim, at the import graph. The runtime version of this claim is in
    `test_torch_free.py`, which runs the command with the stack made unimportable.
    """
    heavy = {"torch", "transformers", "accelerate", "optuna", "datasets",
             "bitsandbytes", "sentencepiece", "gguf", "numpy", "pyarrow", "pandas", "scipy"}
    pkg = CHECKER / "src" / "senbonzakura_check"
    offenders = [f"{p.relative_to(ROOT)}:{line} imports {name}"
                 for p, line, name in _imports_of(pkg)
                 if name.split(".")[0] in heavy]
    assert not offenders, "\n  ".join(["the checker reaches the deep-learning stack:", *offenders])


def test_the_checker_imports_cleanly_in_a_subprocess_of_its_own():
    """Imported on its own rather than after the test suite has already pulled the world in.

    In-process assertions about an import graph are read off the source; this actually does the
    import with nothing else arranged, which is the closest the suite gets to a fresh install.
    """
    got = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True, text=True, check=False, timeout=120)
    assert got.returncode == 0, got.stderr
    assert "'senbonzakura'" not in got.stdout, (
        f"importing the checker dragged in the abliterator: {got.stdout}")


# ── the two versions, which must agree ───────────────────────────────────────────────────────

def test_both_distributions_report_the_same_version():
    """TWO DISTRIBUTIONS OUT OF ONE COMMIT REPORTING DIFFERENT VERSIONS makes a bug report
    untraceable to a build, which is the defect the single-source version rule was written for.

    They are separate files rather than one imported from the other, because reading it across
    the boundary would be an upward import for the sake of one string. So the agreement is kept
    by this test instead.
    """
    from senbonzakura_check._version import __version__ as small

    from senbonzakura._version import __version__ as big

    assert big == small, (
        f"senbonzakura is {big} and senbonzakura-check is {small}. Bump both, or a bug report "
        f"against one cannot be traced to the commit that built the other.")


def test_the_console_scripts_do_not_collide(checker_cfg, main_cfg):
    """With both installed, `senbonzakura check` and `senbonzakura-check` are the same command
    reached two ways. Two distributions claiming one script name is a file conflict.
    """
    assert set(main_cfg["project"]["scripts"]) & set(checker_cfg["project"]["scripts"]) == set()


def test_the_two_distributions_own_disjoint_import_packages(checker_cfg):
    """The reason the code moved out of `senbonzakura/` at all.

    Two distributions installing files into one import package is a conflict: pip permits it,
    uninstalling either breaks the other, and neither owns the directory.
    """
    import senbonzakura_check

    import senbonzakura

    big = Path(senbonzakura.__file__).resolve().parent
    small = Path(senbonzakura_check.__file__).resolve().parent
    assert big != small
    assert not str(small).startswith(str(big) + "/"), (
        "the checker's files live inside the abliterator's package, so the two distributions "
        "would fight over the same directory")


# ── the licence travels with it ──────────────────────────────────────────────────────────────

def test_the_checker_ships_its_own_licence(checker_cfg):
    """AGPL section 4 wants the licence with the work, and "the other distribution has it" is
    not a copy the recipient of this one received.
    """
    assert checker_cfg["project"]["license"] == "AGPL-3.0-or-later"
    assert "LICENSE" in checker_cfg["project"]["license-files"]
    licence = CHECKER / "LICENSE"
    assert licence.is_file() and not licence.is_symlink(), (
        "the checker's LICENSE must be a real file: a symlink does not reliably survive an sdist")
    assert licence.read_text(encoding="utf-8") == (ROOT / "LICENSE").read_text(encoding="utf-8")


def test_the_checker_has_a_readme_that_says_what_it_costs():
    """It is the PyPI project page for this distribution, and the one sentence a reader needs is
    that it pulls nothing.
    """
    readme = (CHECKER / "README.md").read_text(encoding="utf-8")
    assert "pip install senbonzakura-check" in readme
    assert "No torch" in readme or "no torch" in readme


# ── every install site installs the checker first ────────────────────────────────────────────

#: Files that install this project. Enumerated by glob rather than listed, so a new workflow or a
#: new image is covered the day it lands rather than the day somebody remembers this file.
INSTALL_SITE_GLOBS = ("Dockerfile*", ".github/workflows/*.yml", "tools/*.sh")

def _arguments(segment):
    """The whitespace-separated arguments of a shell segment, with quoting stripped.

    ARGUMENTS, NOT THE WHOLE LINE, and the distinction is the point. The first version of this
    check asked whether the line mentioned anything that was not this project and skipped it if
    so; `pip install "requests>=2" ".[abliterate]"` installs both, so the real CUDA bug passed
    the check written to catch it. A test that reads a line as one thing cannot see a line that
    does two.
    """
    return [argument.strip("\"'") for argument in segment.split()]


def _installs_this_project(segment):
    """True when the segment pip-installs the senbonzakura distribution from this tree."""
    if "pip install" not in segment:
        return False
    return any(argument == "." or argument.startswith(".[") or "$WHEEL" in argument
               for argument in _arguments(segment))


def _installs_the_checker(segment):
    """True when the segment pip-installs the checker's tree or its built wheel."""
    if "pip install" not in segment:
        return False
    return any(argument.rstrip("/").endswith("checker") or "CHECK_WHEEL" in argument
               for argument in _arguments(segment))


def test_every_site_that_installs_this_project_installs_the_checker_first():
    """`senbonzakura` declares a dependency on `senbonzakura-check`, and that name is not on
    PyPI. Until it is, every install has to be handed the checker's tree first or the resolver
    goes to the index and finds nothing.

    THE FAILURE THIS EXISTS TO PREVENT, which has already happened twice in one day: the
    dependency was added, ten of twelve CI jobs went red, every `pip install` in the workflow
    was fixed, and the two Dockerfiles were missed; then one Dockerfile was fixed and its CUDA
    twin was missed. Each fix was correct and each was applied to the file that happened to be
    open. A list of the sites is the thing nobody was holding.
    """
    missed = []
    for pattern in INSTALL_SITE_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            checker_seen = False
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                # Split on `&&` because a single RUN line chains several installs, and their
                # order within the line is exactly what is being asserted.
                for segment in line.split("&&"):
                    if _installs_the_checker(segment):
                        checker_seen = True
                    elif _installs_this_project(segment) and not checker_seen:
                        missed.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not missed, (
        "these install the abliterator without installing the checker first, so the resolver "
        "will look for `senbonzakura-check` on an index that does not carry it:\n  "
        + "\n  ".join(missed)
        + "\n\nAdd `pip install --no-deps ./checker` (and, in an image, the `COPY checker/`) "
          "above the line. Delete this test once `senbonzakura-check` is published.")
