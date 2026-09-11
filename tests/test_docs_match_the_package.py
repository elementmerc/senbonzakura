# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The install pages and the command reference, held against what the package actually declares.

WHY THIS FILE EXISTS

Two findings from the 2026-09-10 panel, both of the same shape: a public page describing a
product that had moved underneath it, with nothing comparing the two.

**The install surfaces.** `4ad2f6b` moved torch, transformers, accelerate and optuna back into
`[project].dependencies` and reduced the `abliterate` extra to an alias for the package itself.
`README.md` went on saying `pip install senbonzakura` was "Small, no GPU needed" at "about 210 MB",
and telling a reader to `pip install 'senbonzakura[abliterate]'` to unlock the editor. README.md is
`readme = "README.md"` in pyproject, so that is the PyPI project page. `docs/guide/install.md` was
stale in every claim on its first thirty lines, and one sentence contradicted the CHANGELOG
outright on the first question a new user asks.

**The command reference.** `docs/reference/cli.md` calls itself "the map" and documented
`bench stage` and `bench report`, which do not exist; the family is `head-to-head`. Worse,
`senbonzakura bench --help` does not error, because argparse falls through to the default
abliterator and prints its own help, so a reader who tried it got a plausible screen and no signal.
The same page omitted eleven of the twenty commands, nine of which the CHANGELOG advertises as
"what you can now run".

Per baseline 4, every finding earns a check so the class cannot escape silently next time. These
are those checks.
"""
import re
import sys
from pathlib import Path

import pytest

# NOT `import tomllib`: it entered the standard library in 3.11 and this project declares
# `requires-python = ">=3.10"`. `tests/tomlread.py` exists for exactly this, and its docstring
# records four test files having made the same mistake before. This was the fifth, and CI's 3.10
# job caught it at collection while every interpreter on the machine it was written on was 3.14.
from tomlread import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
BASE_DEPS = PYPROJECT["project"]["dependencies"]
EXTRAS = PYPROJECT["project"].get("optional-dependencies", {})

#: Every page that tells a reader how to install this. README.md is the PyPI project page.
INSTALL_SURFACES = ["README.md", "docs/guide/install.md", "docs/guide/quickstart.md"]


def _text(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _base_package_names():
    return {re.split(r"[<>=!~\[ ]", d, maxsplit=1)[0].strip().lower() for d in BASE_DEPS}


# ── the install surfaces ─────────────────────────────────────────────────────────────

def test_the_heavy_dependencies_really_are_base_dependencies():
    """The premise every assertion below rests on. If this changes, the pages must change too."""
    names = _base_package_names()
    for want in ("torch", "transformers", "accelerate", "optuna"):
        assert want in names, f"{want} is no longer a base dependency; the install pages say it is"


@pytest.mark.parametrize("page", INSTALL_SURFACES)
def test_no_install_page_tells_the_reader_to_add_an_extra_that_is_an_alias(page):
    """`abliterate` resolves to the package itself, so installing it changes nothing.

    A reader following that instruction believes they have unlocked the editor and has done
    nothing at all, which is worse than no instruction.
    """
    alias = {e for e, deps in EXTRAS.items()
             if [d.strip().lower() for d in deps] == ["senbonzakura"]}
    # An INSTRUCTION to install it, not a mention of it. The install page legitimately explains
    # that the extra still resolves and now installs what a bare install does; what it must not
    # do is tell somebody to type it in order to get something they would not otherwise have.
    instructions = [ln for ln in _text(page).splitlines() if "pip install" in ln]
    for extra in alias:
        for line in instructions:
            assert f"[{extra}]" not in line, (
                f"{page} tells the reader to run `{line.strip()}`, and the `{extra}` extra is an "
                f"alias for the package itself, so it installs nothing new")


@pytest.mark.parametrize("page", INSTALL_SURFACES)
def test_install_instructions_name_the_package_not_a_checkout(page):
    """`pip install .` and `pip install '.[hub]'` only work for somebody who already cloned.

    That is the contributor-shaped instruction `a7cd0bb` set out to remove from the user pages,
    and it survived in the extras sections.
    """
    for line in _text(page).splitlines():
        if "pip install" not in line or "senbonzakura" in line:
            continue
        assert not re.search(r"pip install\s+['\"]?\.(\[|['\"]|$)", line), (
            f"{page} tells a stranger to install from a checkout they do not have: {line.strip()}")


@pytest.mark.parametrize("page", INSTALL_SURFACES)
def test_no_install_page_claims_a_bare_install_is_small(page):
    """The figures that were wrong by about seven times, and the words around them.

    Scoped to the sizes actually claimed rather than to any number, so a page may still say how
    big the install IS; what it may not do is describe it as the small one.
    """
    text = _text(page)
    for wrong in ("210 MB", "Small, no GPU needed", "no GPU needed"):
        assert wrong not in text, (
            f"{page} still describes the bare install as it was before torch became a base "
            f"dependency: {wrong!r}")


@pytest.mark.parametrize("page", ["README.md", "docs/guide/install.md"])
def test_every_install_surface_carries_the_windows_warning(page):
    """The caveat that costs a Windows user a day, which existed on exactly one page.

    PyPI's Windows torch is CPU-only, so a laptop with a card installs a torch that cannot see it
    and the search runs on CPU. A caveat only the quickstart carries does not protect a reader who
    arrived on GitHub or in the guide.
    """
    text = _text(page)
    assert "Windows" in text and "idle" in text, (
        f"{page} does not warn that PyPI's Windows torch is CPU-only, so a card there sits idle")
    assert "senbonzakura setup" in text, f"{page} does not name the command that fixes it"


def test_the_migration_note_does_not_contradict_the_changelog():
    """Two published artefacts disagreeing on the first question a new user asks."""
    install = _text("docs/guide/install.md")
    assert "nothing else changed" not in install, (
        "the install page claimed the scoring commands moved behind an extra and nothing else "
        "changed, while the CHANGELOG said the opposite")


# ── the command reference ────────────────────────────────────────────────────────────

def _documented_commands():
    text = _text("docs/reference/cli.md")
    return {m.group(1) for m in re.finditer(r"^\| `([a-z][a-z-]*)(?: [a-z-]+)?`", text, re.MULTILINE)}


def _real_commands():
    from senbonzakura import entry
    # `abliterate` is the default path rather than a delegated module, and `kageyoshi` / `auto`
    # are its presets, so they are named here rather than discovered.
    return set(entry.DELEGATED) | {"abliterate", "kageyoshi", "auto"}


def test_the_reference_documents_no_command_that_does_not_exist():
    """`bench stage` and `bench report` were on the page and nowhere in the parser.

    And argparse does not refuse an unknown first word here: it falls through to the default
    abliterator and prints its help, so the reader gets a plausible screen and no signal.
    """
    invented = _documented_commands() - _real_commands()
    assert not invented, f"docs/reference/cli.md documents commands that do not exist: {invented}"


def test_the_reference_documents_every_command_that_does_exist():
    """It calls itself "the map" and omitted eleven of twenty."""
    missing = _real_commands() - _documented_commands()
    assert not missing, f"docs/reference/cli.md is missing: {sorted(missing)}"
