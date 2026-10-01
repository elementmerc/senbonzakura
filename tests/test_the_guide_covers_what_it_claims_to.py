# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Three documentation gaps the CLI surface sweep found, and the guards that keep them shut.

WHAT PROMPTED IT, 2026-10-01

Three items from the sweep, all the same shape: a command or a licence fact that the tool has and
the documentation does not.

1. **`gate` and `baseline` had no worked example anywhere.** `docs/guide/gating.md` did not exist.
   Two commands that exist to be wired into somebody's CI, with no page showing the wiring.
2. **The `--capability-*` family and `--slow-probe-ok` appeared in no guide page.** A sweep of
   `docs/` and `man/` for `slow-probe-ok` returned nothing. This item was raised by the 2026-09-25
   panel and again by a seven-persona pass, which makes it the longest-deferred of the three.
3. **The AGPL's section 13 reached the README and nothing else.** A sweep of every page under
   `docs/` for "section 13", "network clause", "Affero" and "remote network interaction" returned
   nothing, and the commercial reader who most needs that fact lands on the docs site.

WHY A TEST FOR PROSE

Because the #19 lesson in this sweep was that the fix for a documentation gap is the guard, not
the paragraph. `REPRODUCING.md` vouched for a page carrying a source note that the page did not
carry, and nothing checked. A paragraph added today with nothing holding it is a paragraph that
goes stale on the next rename, silently, and documentation drift has no failing test to announce
it.

So these read the parser and the package for the things that have to be documented, and then look
for them in the pages. A flag renamed in `parser.py` fails here rather than quietly leaving the
guide describing a flag the tool no longer has.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GUIDE = ROOT / "docs" / "guide"
REFERENCE = ROOT / "docs" / "reference"


def _text(path):
    return path.read_text(encoding="utf-8")


def _all_doc_text():
    r"""Every published page a reader can reach, excluding the build output.

    `docs/.vitepress/dist` holds generated copies of every page, so a search that included it
    would find anything at least twice and would keep finding a deleted page until the next build.

    THE MAN PAGE IS IN HERE TOO, because the ledger item behind these tests names
    `man/senbonzakura.1` alongside `docs/guide/`, and somebody reaching for `man senbonzakura` is
    a reader like any other. Its roff markup escapes a hyphen as `\-`, so the text is
    un-escaped before it is searched; without that, every flag name misses.
    """
    out = {}
    for path in sorted((ROOT / "docs").rglob("*.md")):
        rel = path.relative_to(ROOT).as_posix()
        if "/.vitepress/" in f"/{rel}" or "/node_modules/" in f"/{rel}":
            continue
        out[rel] = _text(path)
    man = ROOT / "man" / "senbonzakura.1"
    if man.is_file():
        out["man/senbonzakura.1"] = _text(man).replace("\\-", "-")
    return out


# ── 1. gate and baseline have a worked example ──────────────────────────────────
GATING = GUIDE / "gating.md"


def test_the_gating_page_exists():
    assert GATING.is_file(), (
        "docs/guide/gating.md is gone. `gate` and `baseline` exist to be wired into a build, and "
        "a command with no worked example is a command nobody wires in")


@pytest.mark.parametrize("command", ["senbonzakura baseline", "senbonzakura gate"])
def test_the_gating_page_shows_each_command_being_run(command):
    assert command in _text(GATING), f"{command} is not shown being run on its own page"


def test_the_gating_page_shows_both_verdicts_and_not_only_the_happy_one():
    """A page showing only a pass teaches nobody what a failure looks like.

    The failure output is the whole product here: it is what somebody reads in a build log at
    the moment they care.
    """
    text = _text(GATING)
    assert "gate OK" in text, text[:200]
    assert "gate FAIL" in text
    assert "gate REFUSED" in text, (
        "the page does not show the third verdict. A refusal is not a pass and not a regression, "
        "and a CI step that treats it as either is the defect this page should prevent")


def test_the_gating_page_names_all_three_exit_codes():
    """The exit-code lesson, applied to a page rather than to a check.

    A page saying "exits non-zero" under-specifies a command with two distinct non-zero statuses,
    and the reader writes `gate || echo regression`, which reports a mismatched baseline as a
    regression.
    """
    text = _text(GATING)
    for status in ("| 0 |", "| 1 |", "| 2 |"):
        assert status in text, f"exit status row {status} is missing from the page"


def test_the_gating_page_lists_every_field_that_makes_two_figures_incomparable():
    """Read off `baseline.PINNED`, so adding a pinned field fails here.

    A page listing seven of eight conditions tells a reader the eighth does not matter.
    """
    from senbonzakura import baseline

    text = _text(GATING)
    missing = [field for field in baseline.PINNED if f"`{field}`" not in text]
    assert not missing, (
        f"these fields are pinned in a baseline and are not on the page: {missing}. A reader who "
        f"hits a refusal naming one of them has nowhere to look it up")


def test_the_gating_page_says_what_a_pass_does_not_mean():
    """The command prints this disclaimer every time, so the page cannot be quieter than the tool."""
    text = _text(GATING)
    assert "It is not a statement that the model is safe" in text, (
        "the page drops the caveat the command itself prints on every run")


def test_the_gating_page_is_reachable():
    """A page nothing links to is a page nobody finds, which is the gap this closed.

    Both surfaces, because they are maintained separately: the sidebar in the VitePress config
    and the contents list at the guide root, which that file's own prose says carry the same
    links in the same order.
    """
    config = _text(ROOT / "docs" / ".vitepress" / "config.mjs")
    assert "'/guide/gating'" in config, "the sidebar does not link the gating page"
    assert "/guide/gating" in _text(GUIDE / "index.md"), (
        "the guide's contents page does not link the gating page, and its own prose says it "
        "carries the same links as the sidebar")


# ── 2. the capability family is documented ──────────────────────────────────────
def _abliterate_flags():
    """Every long option the full abliterate parser declares, as the user types it."""
    from senbonzakura.parser import build_parser

    out = set()
    for action in build_parser()._actions:
        out.update(o for o in action.option_strings if o.startswith("--"))
    return out


#: The flags this file insists are documented somewhere a reader can find. Derived rather than
#: listed: the point is that the whole family is covered, and a sixth member added next year is
#: covered by the same rule.
#: Named alongside the family in the ledger item, and the same shape of flag: an override for a
#: refusal the tool raises about its own measurement being worthless. Undocumented for the same
#: reason and fixed in the same pass.
_ALSO = ("--slow-probe-ok", "--short-budget-ok")


def _capability_family():
    return sorted(f for f in _abliterate_flags()
                  if f.startswith("--capability-") or f in _ALSO)


def test_the_capability_family_is_still_what_this_test_thinks_it_is():
    """A guard on the guard. If the prefix changed, the test below would find nothing and pass."""
    family = _capability_family()
    assert len(family) >= 6, (
        f"only {len(family)} flags matched the capability family, so either they were renamed or "
        f"this test's prefix no longer matches: {family}")
    for flag in _ALSO:
        assert flag in family, f"{flag} is no longer a flag on abliterate, or was renamed"


@pytest.mark.parametrize("flag", _capability_family())
def test_every_capability_flag_appears_in_the_documentation(flag):
    """`--help-all` showing a flag is not the same as the documentation explaining it.

    Searched across all of `docs/` rather than one page, because which page owns a flag is a
    judgement and where it is written down is not.
    """
    pages = [rel for rel, text in _all_doc_text().items() if flag in text]
    assert pages, (
        f"{flag} is a real flag on `senbonzakura abliterate` and appears on no documentation "
        f"page. It is hidden from `senbonzakura --help` as well, so a reader has only "
        f"`--help-all` to find it in")


def test_the_slow_probe_refusal_is_shown_rather_than_only_mentioned():
    """It is the one in this family a reader meets as a wall of text mid-run.

    Shown with its three answers, because the flag is the worst of them and a page that names
    only the flag teaches somebody to wait four hours.
    """
    text = _text(REFERENCE / "flags.md")
    assert "--slow-probe-ok" in text
    assert "--load-in-4bit" in text, (
        "the page shows the refusal without the first thing it suggests, which is the answer that "
        "makes the run fast rather than the one that makes it long")


def test_the_capability_page_does_not_confuse_the_two_spellings():
    """`capability` spells the same settings without the prefix, which is a real trap.

    The refusal quoted on the page says `--n 40 --max-new 256`, and those are not flags on
    `abliterate`. A page quoting them without saying so sends a reader to type a flag that does
    not exist.
    """
    text = _text(REFERENCE / "flags.md")
    assert "--n` and `--max-new`" in text or "`--n` and `--max-new`" in text, (
        "the page quotes the capability command's own flag spellings without explaining that "
        "they differ from the prefixed ones used inside an abliterate run")


# ── 3. the AGPL network clause reaches the docs site ────────────────────────────
#: Any one of these means the clause is explained rather than merely linked. Several spellings
#: because the fact matters and the wording is somebody's to choose.
SECTION_13 = ("section 13", "network clause", "Affero", "remote network interaction")


def test_the_network_clause_is_explained_somewhere_on_the_docs_site():
    """The README had it and the docs site did not, and the docs site is where readers land."""
    hits = {rel: [m for m in SECTION_13 if m.lower() in text.lower()]
            for rel, text in _all_doc_text().items()}
    found = {rel: markers for rel, markers in hits.items() if markers}
    assert found, (
        "no page under docs/ mentions the AGPL's section 13 in any of these spellings: "
        f"{SECTION_13}. It is the one licence fact a commercial reader most needs, and it was "
        f"stated nowhere but the licence file until 2026-10-01")


def test_the_clause_is_explained_rather_than_named():
    """Naming a clause number is not explaining it. The obligation has to be in the sentence."""
    text = " ".join(_all_doc_text().values()).lower()
    assert "modified source" in text, (
        "section 13 is named on the docs site without the obligation it imposes being stated: "
        "that a modified version run as a network service has to offer its source to the people "
        "using it over that network")


def test_the_page_says_who_the_clause_does_not_reach():
    """The half that stops the fact being frightening rather than useful.

    Running it privately, however commercially, triggers nothing. A page that states the
    obligation without that line tells an internal user they have a problem they do not have.
    """
    text = " ".join(_all_doc_text().values()).lower()
    assert "internal" in text or "own machine" in text, (
        "the docs state the section 13 obligation without saying it does not reach somebody "
        "running the tool privately, which is most readers")


def test_the_non_commercial_track_is_named_on_the_same_surface():
    """Three licences are in play and two of them bite a commercial reader.

    A page that explains the AGPL and omits that the bundled corpus is CC BY-NC sends somebody
    away believing they have read the licensing.
    """
    text = _text(GUIDE / "what-it-is.md")
    assert "CC BY-NC" in text, text[-400:]
    assert "--track" in text, (
        "the page says the bundled track is non-commercial without naming the flag that replaces "
        "it, which is the only actionable part of that fact")


def test_the_licence_section_matches_the_packaged_expression():
    """The page's claims about the code's licence are read off `pyproject.toml`.

    Not a style check: the expression was corrected once (Q-41, 2026-09-28) and a page naming the
    old one would be worse than a page naming none, because it would read as authoritative.
    """
    # NOT `import tomllib`: it entered the standard library in 3.11 and this project declares
    # `requires-python = ">=3.10"`. `tests/tomlread.py` exists for exactly that, and its own
    # docstring records several test files having made the mistake first.
    from tomlread import tomllib

    data = tomllib.loads(_text(ROOT / "pyproject.toml"))
    expression = str(data["project"].get("license", "")) or ""
    text = _text(GUIDE / "what-it-is.md")
    for part in re.findall(r"[A-Za-z0-9.\-]+-[0-9][A-Za-z0-9.\-]*", expression):
        if part.lower().startswith(("agpl", "cc-by")):
            spelled = part.replace("CC-BY-NC-4.0", "CC BY-NC 4.0")
            assert spelled in text or part in text, (
                f"the packaged licence expression names {part} and the page does not mention it")


@pytest.mark.parametrize("flag", _capability_family())
def test_every_capability_flag_is_in_the_man_page_too(flag):
    """The ledger item names `man/senbonzakura.1` as well as `docs/guide/`.

    Somebody who installed a wheel and typed `man senbonzakura` is a reader like any other, and
    the man page is the one surface that works with no network at all.
    """
    man = _all_doc_text().get("man/senbonzakura.1")
    assert man, "man/senbonzakura.1 is missing, and it is a documented surface"
    assert flag in man, f"{flag} is not in the man page"


def test_the_man_page_still_formats():
    """A roff page with a broken macro renders as nothing in the middle of a section.

    Run through `groff` with warnings fatal where it is available, and skipped with a reason
    where it is not, rather than asserting on the markup by eye.
    """
    import shutil
    import subprocess

    groff = shutil.which("groff")
    if not groff:
        pytest.skip("groff is not installed, so the man page cannot be rendered here")
    man = ROOT / "man" / "senbonzakura.1"
    r = subprocess.run([groff, "-man", "-Tutf8", "-ww", str(man)],
                       capture_output=True, text=True, timeout=60, check=False)
    assert r.returncode == 0, r.stderr
    assert not r.stderr.strip(), f"groff warned about the man page:\n{r.stderr}"
    # And it actually produced the page, rather than exiting 0 over an empty file, which is the
    # shape this sweep keeps finding: a zero status over a check that ran on nothing.
    assert "SENBONZAKURA" in r.stdout.upper(), "groff exited 0 and rendered nothing"


@pytest.mark.parametrize("flag", _capability_family())
def test_every_capability_flag_is_in_the_markdown_docs_and_not_only_the_man_page(flag):
    """Two surfaces, and the man page alone is not enough.

    The ledger item names both, and they reach different people: `man senbonzakura` works offline
    on an installed wheel, and the docs site is where a search engine sends somebody. A test that
    accepted either would have passed on the man page alone, which is how a half fix reads as a
    whole one.
    """
    pages = [rel for rel, text in _all_doc_text().items()
             if rel.endswith(".md") and flag in text]
    assert pages, f"{flag} is in no Markdown page under docs/, only in the man page at best"


def test_the_short_budget_override_explains_why_the_number_would_be_wrong():
    """The flag is easy to reach for and the reason is counter-intuitive.

    A refusal the model never reaches is not counted, so a short budget makes the refusal rate
    look better and the search then prefers configurations whose refusal lands after the cutoff.
    A page naming the flag without that is a page that helps somebody produce a wrong number
    faster.
    """
    text = _text(REFERENCE / "flags.md")
    assert "--short-budget-ok" in text
    assert "never reaches is not counted" in text, (
        "the page offers the override without the mechanism that makes the resulting number "
        "meaningless")
    assert "--length-sweep" in text, (
        "the page does not name the command that answers the question properly, so the override "
        "reads as the only option")


# ── the exit-code trap, which is the defect pointed at a reader ──────────────────
def test_the_gating_page_names_the_two_wrong_lines_as_wrong():
    """A table of statuses does not stop somebody writing `gate || echo regression`.

    The table says what the three codes mean. It does not say that the obvious shell idiom
    collapses them, and the whole reason the trap is worth documenting is that both wrong lines
    look right. So the page shows them and says what each one does to a reader.
    """
    text = _text(GATING)
    assert "Two lines not to write" in text, text[-300:]
    assert "|| echo" in text, "the page does not show the || idiom that mislabels a refusal"
    assert "if senbonzakura gate" in text, (
        "the page does not show the `if cmd; then` idiom that hides why the build is red")


def test_the_gating_page_offers_a_shape_that_cannot_lie():
    """Naming the trap without offering the alternative leaves the reader with the trap.

    The `case` block is the shortest thing that reads the status rather than its truthiness, so
    the page has to carry one; a page that says "be careful" and stops is advice nobody can act
    on.
    """
    text = _text(GATING)
    assert "status=$?" in text, "the page shows no way to read the status itself"
    for code in ("  0)", "  1)", "  2)"):
        assert code in text, f"the worked branch does not handle status {code.strip(') ')}"
    assert "  *)" in text, (
        "the branch has no fallback, so a fourth status this command grows later would fall "
        "through as a pass, which is the same defect one version on")


def test_the_three_statuses_agree_across_every_surface_that_names_them():
    """`gate.py`, `docs/reference/cli.md` and the gating page all state them, so all three must agree.

    CORRECTION TO THIS TEST'S OWN FIRST PREMISE, 2026-10-01: the gating page was written in the
    belief that nothing had named the third status for a reader. `docs/reference/cli.md` already
    did, in its command table, and so did `gate.py`'s module docstring. The page is the first
    WORKED treatment and was not the first mention, and getting that wrong is exactly the kind of
    thing a test should stop somebody repeating. Hence this: the numbers are read from the code
    and checked against both pages.
    """
    from senbonzakura import gate

    assert (gate.OK, gate.REGRESSED, gate.REFUSED) == (0, 1, 2), (
        "the status constants moved; every page that names them needs re-reading")
    cli_md = _text(REFERENCE / "cli.md")
    assert "Exits 0 when steady, 1 on a regression, and 2 when it refused" in cli_md, (
        "the command reference no longer states gate's three statuses, so the gating page is now "
        "the only surface carrying them")
    page = _text(GATING)
    for status in (gate.OK, gate.REGRESSED, gate.REFUSED):
        assert f"| {status} |" in page, f"the gating page's table lost status {status}"


# ── the man page's exit statuses, which documented only 0 ───────────────────────
def test_the_man_page_documents_more_than_exit_status_zero():
    """A sweep item, and the same exit-code defect in the oldest surface in the project.

    `.SH EXIT STATUS` listed `0` and nothing else, on a tool whose commands return three
    different statuses on purpose. A reader of that page could only conclude non-zero means
    trouble, which is the reading that collapses "it regressed" into "it never ran".
    """
    man = _all_doc_text()["man/senbonzakura.1"]
    section = man[man.index(".SH EXIT STATUS"):man.index(".SH ENVIRONMENT")]
    for status in ("0", "1", "2"):
        assert f".B {status}\n" in section, f"exit status {status} is not documented"


def test_the_man_page_says_the_two_status_scales_differ():
    """Measured rather than assumed, and the measurement is the finding.

    `gate`, `check` and `prereg` use 1 for a bad verdict and 2 for no verdict. `doctor` is
    ordered by severity: 0 clean, 1 advisories only and the install works, 2 the install cannot
    do what it claims. Both are deliberate and documented in their own modules, and nothing told
    a user they are different scales, so a `case` block written for one misreads the other in the
    damaging direction: it treats a working install with advisories as a hard failure.
    """
    man = _all_doc_text()["man/senbonzakura.1"]
    assert "THE TWO SCALES ARE NOT THE SAME" in man, (
        "the man page documents three statuses as if every command used them the same way")
    assert "doctor" in man[man.index("THE TWO SCALES"):], "the divergence is named without its case"


def test_the_divergence_the_man_page_describes_is_the_one_in_the_code():
    """The page's claim about `doctor` read straight out of `doctor` and `gate`.

    Without this the page is a paragraph somebody wrote once. With it, a module that changes its
    convention fails here rather than leaving the man page confidently wrong about the thing it
    was added to warn about.
    """
    from senbonzakura import doctor, gate

    assert (doctor.OK, doctor.WARN, doctor.FAIL) == (0, 1, 2)
    assert (gate.OK, gate.REGRESSED, gate.REFUSED) == (0, 1, 2)
    # The divergence itself: doctor's worst outcome shares a number with gate's "I did not
    # compare", and doctor's 1 is an advisory where gate's 1 is a real finding. If that ever
    # stops being true the man page's warning is stale and should be removed, not left.
    assert doctor.FAIL == gate.REFUSED, (
        "doctor's failure status no longer collides with gate's refusal status, so the man page's "
        "warning about the two scales needs re-reading")


def test_the_fourth_status_setup_returns_is_documented_too():
    """`setup` returns 3, and for a while nothing anywhere said so.

    Found 2026-10-01 by driving `tools/ci/docs_commands_run.py` against a working binary rather
    than by reading: `senbonzakura setup` on a machine with no usable stack exits **3**. The man
    page's `EXIT STATUS` had just been completed from "0 only" to three statuses, and it was
    *still* incomplete, because the sweep that completed it read the three commands that share a
    scale and not the one that does not.

    That is the narrower-question defect one more time: a page rewritten to be exhaustive about
    the statuses it knew about. The number is read from `envsetup` here so the page cannot drift
    away from it.
    """
    import re

    source = (ROOT / "src" / "senbonzakura" / "envsetup.py").read_text(encoding="utf-8")
    assert re.search(r"return 3 if verdict == \"blocked\" else 0", source), (
        "`setup` no longer returns 3 for a blocked machine, so the man page's fourth status is "
        "stale and should be removed rather than left standing")
    man = _all_doc_text()["man/senbonzakura.1"]
    section = man[man.index(".SH EXIT STATUS"):man.index(".SH ENVIRONMENT")]
    assert ".B 3\n" in section, "the status `setup` actually returns is still undocumented"
    assert "setup" in section, "status 3 is documented without naming the command that returns it"
