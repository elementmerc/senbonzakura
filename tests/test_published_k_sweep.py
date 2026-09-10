# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The project's most-quoted number, recomputed from data a reader can open.

WHAT WAS WRONG

`p = 0.016` is the multi-direction negative result, and it appears in the README, on the docs
index, in the CHANGELOG and on `docs/guide/what-we-know.md`. The ten per-seed drift values behind
it existed only in `private/results/2026-09-07-k-sweep-read.md`, and `private/` is git-excluded, so
nothing in the published repository let anyone reach the figure. A reviewer on 2026-09-10
reproduced it by recovering those numbers from a working note, which is not a route a reader has.

Two neighbouring pages were corrected for exactly this in the same cycle (`docs/comparison.md` now
says every figure comes from `tools/leak_sweep.py`, with a test comparing the two, and
`docs/guide/benchmark.md` now admits no record of its run is kept). The page carrying the
project's most-quoted claim made no such admission.

These are ten floats and no prompts, so there was never a reason for them not to be committed.
`evidence/k-sweep-2026-08-13/drift-per-seed.json` holds them and this recomputes both published
figures from that file, through the project's own estimator.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from senbonzakura.metrics import permutation_p  # noqa: E402

EVIDENCE = ROOT / "evidence" / "k-sweep-2026-08-13" / "drift-per-seed.json"


@pytest.fixture(scope="module")
def arms():
    doc = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    return doc["drift_kl"]["k1"], doc["drift_kl"]["k2"], doc


def test_the_published_p_value_recomputes_from_the_committed_numbers(arms):
    """0.016, exactly, from the file rather than from a note nobody outside can read."""
    k1, k2, _ = arms
    assert len(k1) == len(k2) == 5
    got = permutation_p(k1, k2, seed=0)
    assert got == pytest.approx(0.015873, abs=5e-6), (
        f"the README, the docs index, the CHANGELOG and what-we-know all quote p = 0.016 and this "
        f"data gives {got:.6f}")
    assert round(got, 3) == 0.016


def test_the_weakened_p_value_recomputes_too(arms):
    """Dropping the worst two-direction seed moves p to 0.048, which the docs also state.

    This is the honest half of the finding and the half a reader is most likely to want to check,
    because it is what turns "twice the collateral damage" into "1.5 to 1.9 times".
    """
    k1, k2, _ = arms
    without_worst = [v for v in k2 if v != max(k2)]
    got = permutation_p(k1, without_worst, seed=0)
    assert round(got, 3) == 0.048, f"the docs say 0.048 and this data gives {got:.6f}"


def test_the_direction_survives_dropping_any_single_seed(arms):
    """The claim the README now makes: the direction is sturdier than the multiplier."""
    k1, k2, _ = arms
    wins = sum(1 for a in k1 for b in k2 if a < b)
    assert wins == 24, (
        f"the README says one direction beats two in 24 of the 25 pairwise seed comparisons, and "
        f"this data gives {wins}")


def test_the_evidence_carries_its_own_caveat(arms):
    """The arms predate the held-out selection, so the second directions were arbitrary.

    An evidence file that records the numbers and not that limitation invites the same over-claim
    the numbers were used for.
    """
    _, _, doc = arms
    assert "caveat" in doc and "ARBITRARY" in doc["caveat"]
    assert doc["run_date"] == "2026-08-13"
    assert doc["prompts_per_score"] == 200


def test_the_evidence_carries_no_prompts():
    """It is ten floats. If it ever grows prompt text, it stops being safe to commit."""
    raw = EVIDENCE.read_text(encoding="utf-8")
    doc = json.loads(raw)
    assert set(doc["drift_kl"]) == {"k1", "k2"}
    assert all(isinstance(v, float) for arm in doc["drift_kl"].values() for v in arm)
