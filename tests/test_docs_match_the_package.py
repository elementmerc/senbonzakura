# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
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


def test_the_pinned_set_states_the_python_it_needs():
    """The numpy release pinned in `constraints.txt` needs Python 3.12; the tool itself supports 3.10.

    So the one command offered to somebody checking our numbers fails on two of the three
    versions this package claims, and pip's message for it names only the package it could not
    find. A reader on 3.10 has no way to tell an unsupported interpreter from a broken pin.

    Found 2026-09-22 because a dependency scanner installed under 3.11 reported our pinned set as
    unresolvable, which read as a defect in the pins until the interpreter was checked.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "docs" / "guide" / "install.md").read_text(encoding="utf-8")
    block = text.split("pip install . -c constraints.txt", 1)
    assert len(block) == 2, "install.md no longer shows the pinned install command"
    assert "3.12 or newer" in block[1][:600], (
        "install.md offers `pip install . -c constraints.txt` without saying it needs Python "
        "3.12 or newer, so a reader on a supported 3.10 meets an unresolvable pin with no reason")


def test_the_install_surfaces_say_the_release_is_not_on_pypi_yet():
    """PANEL FINDING, and it lived in a CI comment until 2026-09-22.

    `senbonzakura` names `senbonzakura-check` as a dependency and that name is not on PyPI
    (verified 2026-09-22: the JSON endpoint returns 404), so the next release cannot be installed
    by anybody. Meanwhile `pip install senbonzakura` resolves to 0.3.0, from July, which the
    CHANGELOG says not to trust. A reader following the front page therefore gets an old version
    silently, and a reader following it after the next release gets a resolver error.

    Our own CI comment stated the problem verbatim, where no user will ever read it. These two
    pages are where a user meets it.

    WHEN THE CHECKER IS PUBLISHED this test should be deleted along with the warnings, and that
    is a deliberate cost: the warnings are wrong the moment the upload succeeds, and a stale
    warning about installability is worse than none.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    # EVERY install surface, not the two somebody happened to think of. `INSTALL_SURFACES`
    # has listed `quickstart.md` all along while this loop named two pages by hand, so the
    # quickstart offered a bare `pip install senbonzakura` with no warning at all until
    # 2026-09-23. Of the three that was the worst one to miss, because it is the page a
    # newcomer lands on, and the list it should have been read from is at the top of this
    # file.
    for name in INSTALL_SURFACES:
        text = " ".join((root / name).read_text(encoding="utf-8").split())
        assert "senbonzakura-check` is not on PyPI yet" in text, (
            f"{name} offers `pip install senbonzakura` without saying the current version cannot "
            f"be installed that way")
        assert "subdirectory=checker" in text, (
            f"{name} says the install is blocked and does not say what to do instead")
        # A COMMAND A STRANGER CAN RUN, which is what the sibling test above is about. The first
        # version of this warning said `pip install ./checker` and that test caught it: a reader
        # who has not cloned cannot follow it, and "the release is broken" is the worst moment to
        # hand somebody a contributor-shaped instruction.
        assert "pip install ./" not in text, (
            f"{name} offers a checkout-only command in the interim install block")


def test_the_readme_leads_with_an_example_a_reader_can_actually_run():
    """PANEL FINDING. The README's only worked example needed `harmful.txt` and `harmless.txt`,
    and sixty lines below it the same page explains that this repository deliberately ships no
    harmful prompt set. So the first command a reader met could not be run by that reader, and
    the runnable one was a link at the bottom of the section.

    The toy track IS committed, so the fix is ordering rather than new machinery: lead with the
    command that works, and keep the real one after it with its input named as something the
    reader supplies.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text(encoding="utf-8")

    toy = readme.find("examples/toy-track/")
    corpus = readme.find("--harmful harmful.txt")
    assert toy != -1, "the README no longer shows the toy-track command"
    assert corpus != -1, "the README no longer shows the real workflow"
    assert toy < corpus, (
        "the README puts the example needing a corpus the reader does not have before the one "
        "that runs on committed data")

    for part in ("examples/toy-track/bad_eval_ds", "examples/toy-track/good_ds"):
        assert (root / part).exists(), f"the README's runnable example names {part}, which is absent"


def test_the_interim_install_says_the_bundled_track_is_not_in_it():
    """SELF-PASS FINDING, 2026-09-22, against an instruction written the same morning.

    The interim install tells a reader to install from the repository URL. The two `.bin` blobs
    under `src/senbonzakura/data/` are generated rather than committed, because they hold harmful
    prompts, so a build from a clone carries neither. The tool then installs, imports and answers
    `--help` exactly as normal and fails on `--track default`, which is the first command the
    quickstart gives.

    This is the same failure that shipped as 0.3.0 and made `--track default` fail for everyone
    who installed it. Writing an install instruction that reproduces it, on the page that warns
    about the release being uninstallable, would have been a poor joke.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    # EVERY install surface, not the two somebody happened to think of. `INSTALL_SURFACES`
    # has listed `quickstart.md` all along while this loop named two pages by hand, so the
    # quickstart offered a bare `pip install senbonzakura` with no warning at all until
    # 2026-09-23. Of the three that was the worst one to miss, because it is the page a
    # newcomer lands on, and the list it should have been read from is at the top of this
    # file.
    for name in INSTALL_SURFACES:
        text = " ".join((root / name).read_text(encoding="utf-8").split())
        assert "track default" in text, f"{name} does not mention --track default at all"
        assert "build_corpora" in text, (
            f"{name} tells a reader to install from the repository without saying the bundled "
            f"corpora are not in it, or how to build them")

    # The premise, asserted rather than assumed: if these ever become tracked, this warning is
    # wrong and should go with them.
    import subprocess
    tracked = subprocess.run(["git", "ls-files", "src/senbonzakura/data/"],
                             capture_output=True, text=True, cwd=root, check=False).stdout
    assert "corpora.bin" not in tracked, (
        "corpora.bin is tracked now, so the warning about a clone install lacking it is stale")


# ── every git reference points at a branch that has this code ────────────────────────


def test_no_documented_git_command_resolves_to_the_default_branch():
    """THE WORKAROUND POINTED AT A SECOND STALE THING.

    Every install page warns that PyPI serves a withdrawn 0.3.0 and tells the reader to install
    from the repository instead. Those commands named no branch, so pip and git took the default
    one, which is `main`. On 2026-09-23 `main` was 593 commits behind `dev` and did not contain
    `checker/` at all, so the documented command failed with "does not appear to be a Python
    project" and the fallback installed something close to the very version the warning was
    about.

    Nothing caught it because nobody had run it: the commands were syntactically fine and pointed
    at a real repository. This asserts the branch is named, which is the part that decides WHICH
    code a stranger gets.

    It will keep being right after `dev` is promoted: `@dev` still resolves then, and if the
    project later wants these to read `@main` deliberately, this fails and asks for that to be a
    decision rather than a default.
    """
    import re as _re

    # The WHOLE token, up to whitespace or a quote. A non-greedy match stopping at the repository
    # name would cut `@dev` off the end and report every fixed line as an offender.
    pattern = _re.compile(r"(git\+https://github\.com/[^\s\"'\\]+|"
                          r"git clone[^\n\"\\]*senbonzakura[^\s\"'\\]*)", _re.IGNORECASE)
    offenders = []
    for rel in [*INSTALL_SURFACES, "notebooks/senbonzakura_colab.ipynb"]:
        path = ROOT / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            fragment = match.group(0)
            named = "@dev" in fragment or "@main" in fragment or "--branch" in fragment
            if not named:
                offenders.append(f"{rel}: {fragment[:90]}")
    assert not offenders, (
        "these documented git commands name no branch, so they resolve to the repository's "
        "default branch rather than to the code the page describes:\n  " + "\n  ".join(offenders))
