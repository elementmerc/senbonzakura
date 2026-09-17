# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The man page is an installed artefact, and it went stale where nobody looks.

FOUND BY A HOSTILE OUTSIDE REVIEW, 2026-09-17. `man/senbonzakura.1` installs to
`share/man/man1` and reaches every user. It was stamped `senbonzakura 0.3.0` against a
`0.4.0.dev9` binary, described an extraction method replaced on 2026-08-03, and made a positive
causal claim the project had already measured as false:

    refusal is not one blade, it is a small subspace, and cutting several
    directions at once removes the stubborn residual without wrecking coherence

Against this project's own five seed result: two directions cost roughly 1.5 to 1.9 times the
collateral damage of one at the same refusal rate, permutation p = 0.016. The docs site retracts
the claim prominently. The manual that ships with the binary did not, and the README is not what
`pip install` puts on somebody's machine.

Three independent hostile reviewers reached the same sentence about this project from different
directions on the same night: the prose is more honest than the product, and the retracted
claims survive in the surfaces that ship. This file is one of the gates for that.
"""
import re
from pathlib import Path

import pytest

MAN = Path(__file__).resolve().parent.parent / "man" / "senbonzakura.1"

pytestmark = pytest.mark.skipif(not MAN.exists(), reason="no man page in this tree")


def _text():
    return MAN.read_text(encoding="utf-8")


def _release_version():
    """The package version with any dev or release-candidate suffix removed.

    The man page is stamped with the RELEASE it documents, so it does not have to be touched on
    every dev cut, and a real drift (0.3.0 against 0.4.0) still fails.
    """
    from senbonzakura import _version
    raw = getattr(_version, "__version__", None) or _version.version
    return ".".join(raw.split(".")[:3]).split("dev")[0].rstrip(".")


def test_the_manual_is_stamped_with_the_version_it_documents():
    """It said 0.3.0 for two releases, which is how the rest of this file's findings survived."""
    found = re.search(r'\.TH\s+SENBONZAKURA\s+1\s+"[^"]*"\s+"senbonzakura ([^"]+)"', _text())
    assert found, "the .TH line does not carry a version at all"
    assert found.group(1) == _release_version(), (
        f"the man page says senbonzakura {found.group(1)} and this package is "
        f"{_release_version()}. A stale stamp is the sign that nothing else on the page was "
        f"reviewed either, which is exactly how it came to document a replaced extractor and a "
        f"withdrawn claim.")


@pytest.mark.parametrize("claim,why", [
    ("without wrecking coherence",
     "the multi-direction benefit claim, measured false at p = 0.016 across five seeds"),
    ("PCA of",
     "the PCA extractor, replaced by clustered difference of means on 2026-08-03"),
])
def test_the_manual_does_not_carry_a_withdrawn_claim(claim, why):
    assert claim not in _text(), (
        f"the installed manual still says {claim!r}: {why}. The docs site retracts it and the "
        f"man page is what pip puts on the user's machine.")


def test_no_shipped_surface_leads_with_the_withdrawn_claim():
    """The claim came out of the README and was meant to come out of every metadata surface.

    Two were missed, and they are the two that `pip install` actually puts on a machine: the
    man page's NAME line and the first sentence of `senbonzakura --help`. A reader who never
    opens the repository meets only those.

    This checks the LEADING claim, not the word. "multi-directional weight ablations" as a
    description of what the search explores is a true statement about a capability and stays.
    """
    import subprocess
    import sys

    name = re.search(r"\.SH NAME\n(.*?)\.SH", _text(), re.S)
    assert name, "the man page has no NAME section"
    assert "multi-direction refusal abliteration" not in name.group(1).lower(), (
        "the man page NAME line still leads with the multi-direction claim")

    out = subprocess.run([sys.executable, "-m", "senbonzakura", "--help"],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL,
                         check=False, timeout=300).stdout
    first = out.split("positional arguments")[0]
    assert not re.search(r"^multi-direction refusal abliteration", first.strip(), re.I | re.M), (
        "`senbonzakura --help` still opens with the multi-direction claim")


def test_the_manual_says_the_multi_direction_question_is_open():
    """Silence would leave a reader with the old claim they half remember.

    Deleting the sentence is not enough. The claim was the headline for two releases, so the page
    has to say it was tested and withdrawn, or a returning reader assumes it still holds and was
    merely trimmed for space.
    """
    text = _text()
    assert "withdrawn" in text, "the manual removes the claim without saying it was withdrawn"
    assert "p = 0.016" in text, "the manual withdraws the claim without the measurement behind it"
