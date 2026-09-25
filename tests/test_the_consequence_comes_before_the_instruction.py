# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""On every public surface, what this does comes before how to do it.

FOUND BY A HOSTILE OUTSIDE REVIEW, 2026-09-17, and the reviewer's own summary is the fairest
statement of it: "Nobody is hiding anything. The material is simply downstream of the
instructions in every surface I checked."

That was true and it was measurable:

  the docs sidebar   "Start here" ran Quickstart, What it is, Install, Your first run, so the
                     page explaining what abliteration destroys sat behind the page telling you
                     how to do it.
  the Quickstart     offered that page in its footer as "the background, if you would rather
                     have it before the buttons", which makes reading it the optional path.
  the landing page   a hero, a tagline and three feature cards, all about measurement quality,
                     with no statement of what the tool produces.
  "Your first run"   no line about licence or danger anywhere on it.
  the README         `pip install` at line 42; the disclosure of roughly 6,200 bundled harmful
                     prompts at line 196; "removes safety guardrails wholesale" at line 203. The
                     README is what PyPI renders, so that ordering is what PyPI shows.

None of that was concealment and all of it was ordering. An ordering is a choice, it was the
wrong one, and a choice that nothing checks drifts back.

WHAT THIS DOES NOT ASSERT. Not the wording, which will change, and not the presence of any
particular sentence for its own sake. It asserts POSITION: on each surface a reader meets first,
the consequence appears before the first instruction. That is the property the review was about.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

#: Words that only appear when a page is saying what the tool produces rather than how to run it.
#: Deliberately a set of alternatives rather than one pinned sentence: the claim is that the
#: consequence is stated, not that it is stated in today's words.
CONSEQUENCE = re.compile(
    r"answer requests the original refused"
    r"|removes safety guardrails"
    r"|refusal behaviour (?:is|has been) removed"
    r"|without saying what it is",
    re.IGNORECASE)

pytestmark = pytest.mark.skipif(not DOCS.is_dir(), reason="no docs tree in this checkout")


def _first_match(pattern, text):
    found = pattern.search(text)
    return found.start() if found else None


class TestTheSidebarLeadsWithWhatItIs:

    def test_what_it_is_comes_before_quickstart(self):
        config = (DOCS / ".vitepress" / "config.mjs").read_text(encoding="utf-8")
        start_here = config.split("'Start here'", 1)[1].split("]", 1)[0]
        what_it_is = start_here.find("what-it-is")
        quickstart = start_here.find("quickstart")
        assert what_it_is != -1 and quickstart != -1, "Start here lost one of its two first pages"
        assert what_it_is < quickstart, (
            "the sidebar puts Quickstart before What it is, so the page explaining what "
            "abliteration destroys sits behind the page telling you how to do it")


class TestEveryPageThatGivesACommandSaysWhatItMakesFirst:

    @pytest.mark.parametrize("page", ["quickstart.md", "first-run.md"])
    def test_the_consequence_precedes_the_first_command(self, page):
        path = DOCS / "guide" / page
        if not path.is_file():
            pytest.skip(f"{page} is not in this checkout")
        text = path.read_text(encoding="utf-8")

        consequence = _first_match(CONSEQUENCE, text)
        assert consequence is not None, (
            f"{page} gives commands and never says what the result of running them is")

        first_command = text.find("```")
        assert first_command != -1, f"{page} has no command block, so this test is on the wrong page"
        assert consequence < first_command, (
            f"{page} shows its first command at character {first_command} and does not say what "
            f"the model will do until character {consequence}. The reader meets the instruction "
            f"before the consequence.")

    def test_the_quickstart_does_not_offer_the_background_as_optional(self):
        """"If you would rather have it before the buttons" made reading it the optional path."""
        text = (DOCS / "guide" / "quickstart.md").read_text(encoding="utf-8")
        offer = re.search(r"if you would rather have it before the buttons", text)
        if offer is None:
            return
        # Kept only where the page is explicitly describing the old wording as the mistake.
        context = text[max(0, offer.start() - 260):offer.start()]
        assert "used to" in context, (
            "the Quickstart still offers What it is as optional background rather than as the "
            "page that comes before it")


class TestTheLandingPageSaysWhatTheToolProduces:

    def test_the_consequence_is_on_the_landing_page_at_all(self):
        text = (DOCS / "index.md").read_text(encoding="utf-8")
        assert CONSEQUENCE.search(text), (
            "docs/index.md is a hero, a tagline and three feature cards about measurement "
            "quality, and never says what the tool produces")


class TestTheReadmeIsWhatPypiRenders:
    """`readme = "README.md"` in pyproject.toml, so this ordering is PyPI's ordering."""

    def test_the_consequence_comes_before_the_install_command(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        install = text.find("pip install senbonzakura")
        consequence = _first_match(CONSEQUENCE, text)
        assert install != -1, "the README no longer shows an install command"
        assert consequence is not None, "the README never says what the tool produces"
        assert consequence < install, (
            f"the README shows `pip install` at character {install} and does not say what the "
            f"model will do until character {consequence}. This file is what PyPI renders.")

    def test_the_bundled_corpus_is_disclosed_before_the_install_command(self):
        """The operator decided on 2026-09-17 that the corpus ships. Disclosure is the condition."""
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        install = text.find("pip install senbonzakura")
        disclosed = text.find("6,200 harmful prompts")
        assert disclosed != -1, "the README no longer discloses the bundled harmful prompts"
        assert disclosed < install, (
            "the bundled corpus is disclosed after the install command, so a reader installs "
            "6,200 harmful prompts and is told afterwards")
