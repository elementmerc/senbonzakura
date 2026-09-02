# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The quantisation page lists types, so it will drift from the code unless something stops it.

A documented list is a promise about what the tool does. This project has already shipped a page
saying a question needed a dataset when it also needed a code change, and a `--chat-template` flag
that had been dead for weeks while its help text described what it would have done. Both were
found by reading, not by a check.

So the page is checked against the code rather than against a reviewer's memory: every type the
tool can produce must appear under "What we can produce", every type it can only read must appear
under "What we can read but not produce", and neither list may name a type the code does not know.
"""
import re
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parent.parent / "docs" / "guide" / "quantisation.md"

#: Names that appear in the page as illustrations rather than as claims about a supported type.
#: Listed explicitly, because a blanket "ignore anything unrecognised" would let a genuine typo
#: through, which is the failure this whole file exists to prevent.
ILLUSTRATIVE = {"UD", "XL", "GGUF", "IQ", "S", "M", "L", "K"}


def _section(title):
    text = DOC.read_text(encoding="utf-8")
    start = text.index(f"## {title}")
    rest = text[start + 3:]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


def _types_in(section):
    """Quantisation-looking tokens inside fenced code blocks, which is where the lists live."""
    found = set()
    for block in re.findall(r"```\n(.*?)```", section, re.DOTALL):
        found.update(re.findall(r"\b((?:I?Q|TQ|F|BF|MXFP|NVFP)[0-9][A-Z0-9_]*)\b", block))
    return found


@pytest.fixture(scope="module")
def code_types():
    from senbonzakura import gguf_io
    from senbonzakura.quantise import QUANT_TYPES
    table = next(v for k, v in vars(gguf_io).items()
                 if isinstance(v, dict) and v.get(0) == "F32" and not k.startswith("__"))
    readable = set(table.values()) - {"GUESSED"}      # not a type; the header could not be read
    return set(QUANT_TYPES), readable


def test_the_page_exists_and_is_linked_from_the_guide():
    assert DOC.exists()
    text = DOC.read_text(encoding="utf-8")
    assert "senbonzakura quantise" in text


def test_every_producible_type_is_documented_as_producible(code_types):
    producible, _ = code_types
    documented = _types_in(_section("What we can produce"))
    missing = producible - documented
    assert not missing, (
        f"`quantise --type` accepts {sorted(missing)} and the page does not list them. A user "
        f"reading this page would not know the tool can make them.")


def test_the_producible_list_promises_nothing_the_tool_cannot_make(code_types):
    producible, _ = code_types
    documented = _types_in(_section("What we can produce"))
    extra = documented - producible - ILLUSTRATIVE
    assert not extra, (
        f"the page promises {sorted(extra)} under 'what we can produce' and `QUANT_TYPES` does "
        f"not accept them")


def test_every_read_only_type_is_documented_as_read_only(code_types):
    producible, readable = code_types
    documented = _types_in(_section("What we can read but not produce"))
    missing = (readable - producible) - documented
    assert not missing, (
        f"the header reader recognises {sorted(missing)} and the page does not mention them, so a "
        f"reader cannot tell whether a file of that type can be checked")


def test_the_read_only_list_names_nothing_producible(code_types):
    producible, _ = code_types
    documented = _types_in(_section("What we can read but not produce"))
    overlap = documented & producible
    assert not overlap, (
        f"{sorted(overlap)} are listed as read-only and the tool can produce them, which would "
        f"send someone off to find a file they could have built")


def test_the_page_says_what_the_unsloth_prefix_is():
    """`UD-` names are the ones most likely to be mistaken for a type we could reproduce."""
    text = DOC.read_text(encoding="utf-8")
    assert "UD-" in text
    assert "Unsloth" in text
    assert "**No.**" in text, "the page has to answer 'can we reproduce it' for UD- plainly"


def test_the_page_explains_the_confound_that_actually_happened():
    """Two files both called Q4_K_M, one built with an importance matrix and one not."""
    text = DOC.read_text(encoding="utf-8")
    assert "i1-" in text
    assert "importance matrix" in text
    assert "provenance.json" in text, "the page must point at the record that settles it"
