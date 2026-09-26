# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The caveat blocks on the docs site have to be readable, and that is measurable.

WHY THIS IS A TEST AND NOT A JUDGEMENT

This project puts its conditions, its withdrawn figures and the paths to its recomputable
artefacts inside `::: tip` and `::: warning` blocks. VitePress tints those with the brand colour
and then colours inline code inside them from the same ramp, so on a dark theme the two drift
together. Measured on 2026-09-26, code inside a caveat block ran at 3.47:1, under the 4.5:1 WCAG
AA asks for body text, and the operator could not read a file path in one.

A contrast failure is the kind of defect that never appears in a diff and never fails a build. It
is also arithmetic, so it can be checked rather than noticed.
"""
import re
from pathlib import Path

import pytest

PALETTE = Path(__file__).resolve().parent.parent / "docs" / ".vitepress" / "theme" / "palette.css"

#: WCAG 2.1 AA for body text. Large text is allowed 3:1; none of this is large text.
AA_BODY = 4.5

#: The dark page background and the brand tint VitePress lays over it for a custom block. Both are
#: read from the stylesheet rather than hardcoded, so a palette change moves the target with it.
BLOCK_TINT_ALPHA = 0.16


def _luminance(hex_colour):
    h = hex_colour.lstrip("#")
    channels = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b):
    la, lb = _luminance(a), _luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _over(fg, alpha, bg):
    f, b = fg.lstrip("#"), bg.lstrip("#")
    return "#" + "".join(f"{round(alpha * int(f[i:i+2], 16) + (1 - alpha) * int(b[i:i+2], 16)):02x}"
                         for i in (0, 2, 4))


def _css():
    return PALETTE.read_text(encoding="utf-8")


def _declared(block_pattern, prop):
    """The value of `prop` inside the first rule whose selector matches, or None."""
    m = re.search(block_pattern + r"\s*\{([^}]*)\}", _css())
    if not m:
        return None
    found = re.search(prop + r"\s*:\s*([^;]+);", m.group(1))
    return found.group(1).strip() if found else None


def test_the_reference_numbers_this_file_depends_on_are_still_in_the_palette():
    """A test reading colours out of a stylesheet goes quiet the moment the selector is renamed."""
    css = _css()
    assert "--vp-c-bg: #12151b" in css, "the dark page background is no longer where this reads it"
    assert ".dark .vp-doc .custom-block code" in css, (
        "the rule that fixes caveat-block contrast is gone. It was added because inline code in "
        "those blocks measured 3.47:1 on the dark theme")


@pytest.mark.parametrize(("selector", "what"), [
    (r"\.dark \.vp-doc \.custom-block code", "inline code in a caveat block"),
    (r"\.dark \.vp-doc \.custom-block a", "a link in a caveat block"),
])
def test_caveat_block_text_meets_wcag_aa_on_the_dark_theme(selector, what):
    colour = _declared(selector, "color")
    assert colour and colour.startswith("#"), (
        f"{what} has no explicit colour, so it inherits the brand ramp that caused this")
    background = _over("#9db0d0", BLOCK_TINT_ALPHA, "#12151b")
    ratio = contrast(colour, background)
    assert ratio >= AA_BODY, (
        f"{what} is {colour} on {background}, which is {ratio:.2f}:1 and under WCAG AA's "
        f"{AA_BODY}:1 for body text. These blocks carry the conditions on the published numbers "
        f"and the paths to the artefacts, so unreadable here is worse than unreadable elsewhere")


def test_the_contrast_maths_is_right():
    """The formula itself, against values whose answers are known.

    A broken helper that returned something large would pass every assertion above while measuring
    nothing, which is the shape this project keeps finding in its own gates.
    """
    assert round(contrast("#ffffff", "#000000"), 1) == 21.0
    assert round(contrast("#ffffff", "#ffffff"), 1) == 1.0
    assert contrast("#6d81a6", "#282e38") < AA_BODY   # the failure this file was written for
