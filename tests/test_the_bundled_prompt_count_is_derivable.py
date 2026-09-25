# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The number of harmful prompts a `pip install` deposits, recomputed rather than asserted.

This figure is the project's central dual-use disclosure. It appears in the README, the acceptable
use policy, the paper, the install guide, the quickstart, the docs index and the evaluation track
card, and it is the sentence a reader is entitled to check.

WHAT WAS WRONG UNTIL 2026-09-25. Every one of those places said "roughly 6,500 harmful prompts".
Recomputing gives 6,228, and 6,500 is reachable only by counting the 250 rows of `xstest-safe`,
which are BENIGN controls, as harmful. Two tests pinned the string and neither recomputed it, so
the suite was asserting that a wrong number stayed wrong.

Three source comments had a different error: `bundled.py`, `tests/test_shipped_files.py` and
`build_capability_probe.py` all carried 4,895, which is the packed track ALONE, while two of them
attributed it to the track and the corpora together. So the tree held one figure that was too
large and another that was too small, and neither was derivable from anything.

This test recomputes from the shipped blobs and the corpus registry, so the prose can only drift
by failing.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: `xstest-safe` is benign by construction: it is the control set for over-refusal, and counting
#: it as harmful is exactly the error this file exists to prevent.
BENIGN_CORPORA = frozenset({"xstest-safe"})

PUBLIC_FILES = [
    "README.md", "ACCEPTABLE-USE.md", "paper.md", "docs/index.md",
    "docs/guide/install.md", "docs/guide/quickstart.md", "docs/evaluation-track-card.md",
]


def _track_harmful():
    from senbonzakura import bundled
    counts = bundled.manifest()["counts"]
    return counts["bad_ds"] + counts["bad_eval_ds"]


def _corpora_split():
    from senbonzakura import corpora
    harmful = benign = 0
    for name, spec in corpora.CORPORA.items():
        n = getattr(spec, "rows", None) or getattr(spec, "n", None) or getattr(spec, "count", None)
        if not n:
            continue
        if name in BENIGN_CORPORA:
            benign += n
        else:
            harmful += n
    return harmful, benign


@pytest.fixture(scope="module")
def totals():
    pytest.importorskip("senbonzakura.bundled")
    from senbonzakura import bundled
    if not bundled.is_available():
        pytest.skip("the packed track is not in this checkout; it is built at release time")
    c_harm, c_benign = _corpora_split()
    return {"track": _track_harmful(), "corpora_harmful": c_harm, "corpora_benign": c_benign}


def test_the_harmful_total_is_what_the_public_files_claim(totals):
    total = totals["track"] + totals["corpora_harmful"]
    claimed = round(total, -2)
    for name in PUBLIC_FILES:
        text = (ROOT / name).read_text(encoding="utf-8")
        found = {int(m.replace(",", "")) for m in re.findall(r"\b(\d,\d00)\b harmful prompts", text)}
        if not found:
            continue
        assert found == {claimed}, (
            f"{name} says {sorted(found)} harmful prompts; the blobs hold {total:,}, which rounds "
            f"to {claimed:,}. The figure is recomputed here rather than pinned as a string, so a "
            f"blob change has to move the prose with it.")


def test_the_benign_controls_are_not_counted_as_harmful(totals):
    """The specific arithmetic that produced the wrong number for months."""
    wrong = totals["track"] + totals["corpora_harmful"] + totals["corpora_benign"]
    right = totals["track"] + totals["corpora_harmful"]
    assert round(wrong, -2) != round(right, -2), (
        "this test can no longer tell the two arithmetics apart, so it has stopped guarding "
        "anything. If the corpora changed, re-derive the figure rather than deleting this.")
    for name in PUBLIC_FILES:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert f"{wrong // 100 * 100:,} harmful" not in text, (
            f"{name} states the figure that counts {totals['corpora_benign']} benign control "
            f"prompts as harmful")


def test_the_readme_shows_the_working(totals):
    """A figure nobody can derive is a figure nobody can check, which is this project's own rule."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{totals['track']:,}" in text, (
        "the README should state the track's own harmful count so the total can be reconstructed")
    assert f"{totals['corpora_harmful']:,}" in text
