# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The three things JOSS asks a project to document, and the leak rule the templates inherit.

WHY THIS EXISTS

`08-v1.0-joss-ready.md` carries the exit-gate line: *"Community guidelines cover all three things
JOSS asks for: how to contribute, how to report issues, how to seek support. A CONTRIBUTING.md
covering only the first does not satisfy the checklist. An issue template and a named support
channel go with it."*

Measured 2026-09-21: `CONTRIBUTING.md` covered contributing, security reporting and wrong-number
reporting, and `.github/` held nothing but `workflows`. No issue template existed at all, so two
of the three were addressed in prose and the third was not addressed anywhere.

THE PART THAT IS NOT PAPERWORK

An issue template is the most-read prose this project ships, because it is the one piece a
stranger meets at the moment they are about to paste something. This repository is public and its
history is permanent, so every template has to carry the same rule the commit gate enforces: do
not paste model generations or the prompts that produced them. A template that asks for "the full
output" is a leak invitation with a form attached, and no pre-commit hook can catch what somebody
types into a web page.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / ".github" / "ISSUE_TEMPLATE"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"


def _templates():
    return sorted(TEMPLATES.glob("*.md")) if TEMPLATES.is_dir() else []


def test_there_is_an_issue_template_directory():
    assert TEMPLATES.is_dir(), (
        f"{TEMPLATES.relative_to(ROOT)} does not exist. JOSS asks for an issue template as one "
        f"of three community-guideline items, and without it the parametrised checks below run "
        f"zero cases and report green.")


def test_at_least_one_template_exists():
    assert _templates(), (
        "the issue template directory exists and holds no templates, so every check below is "
        "vacuous.")


def test_the_wrong_number_template_exists_and_is_listed_first():
    """The distinctive report of this project, which CONTRIBUTING already calls the most valuable
    issue somebody can file. A tool whose pitch is receipts should make disputing a receipt the
    easiest thing on the form.

    Sorting is alphabetical in the GitHub UI by file name, so the assertion is on presence
    rather than order; the ordering claim belongs to the template's own text.
    """
    names = [p.name for p in _templates()]
    assert "wrong-number.md" in names, names


@pytest.mark.parametrize("path", _templates(), ids=lambda p: p.name)
def test_every_template_warns_against_pasting_generations(path):
    """THE RULE NO HOOK CAN ENFORCE.

    `tools/ci/check_prompt_artefacts.py` refuses a COMMIT carrying prompts or generations. It
    cannot see a GitHub issue. The only control on that path is the sentence in front of the
    person about to paste, so every template carries it and this test is what keeps it there.
    """
    text = path.read_text(encoding="utf-8").lower()
    assert "generation" in text and "prompt" in text, (
        f"{path.name} does not warn against pasting model generations or the prompts that "
        f"produced them. This repository is public and its history is permanent, and no "
        f"pre-commit hook can catch what somebody types into a web form.")


@pytest.mark.parametrize("path", _templates(), ids=lambda p: p.name)
def test_every_template_asks_for_the_version(path):
    """A report without a version cannot be reproduced and usually cannot be acted on."""
    assert "version" in path.read_text(encoding="utf-8").lower(), path.name


def test_a_support_route_is_named_somewhere_a_stranger_will_meet_it():
    """The third JOSS item, and the one that was missing entirely.

    Checked in `config.yml` rather than in prose, because that is the file GitHub renders on the
    new-issue page. A support channel documented only in a README section is documented for
    somebody who already knows to look.
    """
    config = TEMPLATES / "config.yml"
    assert config.is_file(), "no .github/ISSUE_TEMPLATE/config.yml, so no support route is shown"
    text = config.read_text(encoding="utf-8")
    assert "contact_links" in text, text[:200]
    assert "security" in text.lower(), (
        "config.yml does not route a security report away from the public tracker. "
        "CONTRIBUTING.md says not to open a public issue, and this is the page where somebody "
        "is about to.")


def test_blank_issues_stay_enabled():
    """A template set that cannot be escaped turns the report nobody anticipated into the report
    nobody files. The reports this project most needs are the ones it did not think to ask for,
    and four of its own worst defects were found by somebody doing something unplanned.
    """
    text = (TEMPLATES / "config.yml").read_text(encoding="utf-8")
    assert "blank_issues_enabled: true" in text


@pytest.mark.parametrize("topic", ["contribut", "security", "wrong number"])
def test_contributing_still_covers_what_it_covered(topic):
    """The templates are an addition, not a replacement. JOSS reads CONTRIBUTING.md first."""
    assert topic in CONTRIBUTING.read_text(encoding="utf-8").lower(), topic


@pytest.mark.parametrize("path", _templates(), ids=lambda p: p.name)
def test_a_template_states_the_rule_rather_than_pointing_at_it(path):
    """THE WEAKNESS IN THE TEST ABOVE, found by a panel reviewer on 2026-09-21.

    That test asserts the words `generation` and `prompt` appear somewhere in the template. A
    template whose only mention was the sentence "do NOT include model generations or the prompts
    that produced them; see CONTRIBUTING.md" satisfied it, and root `CONTRIBUTING.md` carried no
    such rule at all: it lived in the VitePress page of nearly the same name, which is neither the
    file the template names nor the file a JOSS reviewer opens first.

    So the check covered one spelling of the requirement, the presence of two words, and not the
    thing the requirement is for: that a person about to paste meets an instruction rather than a
    redirection. The rule has to be readable without leaving the form.
    """
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "do not" in lowered or "don't" in lowered or "not include" in lowered, (
        f"{path.name} mentions generations without telling anybody not to paste them")

    # A pointer is allowed BESIDE the rule and not INSTEAD of it. If the template names another
    # document, that document has to carry the rule too, or the reader is sent somewhere that
    # does not answer the question they were about to get wrong.
    if "contributing.md" in lowered:
        contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8").lower()
        assert "generation" in contributing and "prompt" in contributing, (
            f"{path.name} points the reader at CONTRIBUTING.md, which does not state the rule. "
            f"A pointer to a document that does not answer the question is worse than no pointer: "
            f"it reads as though the reader has been told.")


def test_the_root_contributing_file_carries_the_rule_itself():
    """Asserted on the ROOT file specifically, because that is the one the templates name, the one
    GitHub links from the issue form, and the one a JOSS reviewer opens. `docs/contributing.md` is
    a different file with a nearly identical name, and the rule living only there is what made the
    pointer above dangle.
    """
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8").lower()
    assert "generation" in text and "prompt" in text
    assert "do not" in text or "don't" in text
    assert "no hook can see a web form" in text, (
        "the root contributing file states the rule without saying why it matters, and the reason "
        "is the whole argument: on the issue path the sentence is the only control there is")
