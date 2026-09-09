# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The published norm-restore range, checked against the script that produces it.

WHY THIS IS A TEST AND NOT A CAREFUL EDIT

`tools/leak_sweep.py` exists because these numbers were once quoted from a handoff and lived
nowhere a machine could reproduce them; its own docstring says a figure drawn from prose is a
picture of somebody's memory. The script was then written, and the page went on quoting figures
that the script does not produce.

`docs/comparison.md` said survival runs "from about 5% ... to 32% at 4x and 46% at 10x". Measured:
the median at 4x is 34.3 to 36.4% across direction counts and at 10x it is 41.1 to 44.1%. So 32%
sat near the bottom of one spread and 46% near the top of another, which makes the curve look
steeper than anything the script has produced. One reviewer said 46% "is produced by nothing at any
spread"; that is very slightly too strong, since 46.2% appears as a MAXIMUM at 10x with four
directions. It is not a median at any spread, which is the claim the page was making.

The sweep runs in about four seconds and is deterministic, so there is no reason for a fixture and
no reason for this page ever to drift again.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "docs" / "comparison.md").read_text(encoding="utf-8")


def medians_at(spread, rows):
    return [r["leak_fraction_median"] * 100 for r in rows
            if r["row_length_spread"] == spread and r["rounds"] == 0]


def test_the_published_range_is_what_the_script_measures():
    import importlib.util
    spec = importlib.util.spec_from_file_location("leak_sweep", ROOT / "tools" / "leak_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = mod.sweep()

    even, mid, wide = medians_at(0.2, rows), medians_at(4.0, rows), medians_at(10.0, rows)
    assert even and mid and wide, "the sweep no longer covers 0.2x, 4x and 10x"

    sentence = PAGE.split("median survival runs from about")[1].split("\n\n")[0]
    # Every number inside a bold span, in order: "**6%**", "**34 to 36%**", "**41 to 44%**"
    # gives 6, 34, 36, 41, 44. Pulling only the first of each span is how the first version of
    # this test read three figures where the page prints five.
    quoted = [float(n) for span in re.findall(r"\*\*([^*]+)\*\*", sentence)
              for n in re.findall(r"[\d.]+", span)]
    assert len(quoted) == 5, f"expected five figures, got {quoted} from: {sentence[:150]}"

    # Every figure the page prints has to sit inside the measured medians for its own spread,
    # rather than being borrowed from the top or the bottom of a different one.
    assert round(min(even)) <= quoted[0] <= round(max(even)) + 1, (
        f"the page says {quoted[0]}% at a 0.2x spread; measured medians are "
        f"{min(even):.1f} to {max(even):.1f}%")
    for value, band, label in ((quoted[1], mid, "4x"), (quoted[2], mid, "4x"),
                               (quoted[3], wide, "10x"), (quoted[4], wide, "10x")):
        assert min(band) - 1 <= value <= max(band) + 1, (
            f"the page says {value}% at {label}; measured medians are "
            f"{min(band):.1f} to {max(band):.1f}%")


def test_the_page_says_where_its_numbers_came_from():
    """A figure with no route back to the thing that produced it is the defect being fixed."""
    assert "tools/leak_sweep.py" in PAGE
    assert "a test compares this page against it" in PAGE
