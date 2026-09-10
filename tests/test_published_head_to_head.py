# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The benchmark page's coherence figures, recomputed from the run's own artefacts.

WHAT THIS EXISTS TO PREVENT, and it is not hypothetical

`docs/guide/benchmark.md` carries a superseding block stating that the 2026-09-10 head-to-head
found no detectable coherence difference. Its first version quoted **p = 0.405** and a
senbonzakura worst seed of **0.0605**. Both were wrong. They came from a synthetic run directory
built earlier the same day to exercise the generalised `verdict()`, with per-seed values
approximated from memory because the real arms were on another machine and unreachable. The report
that fixture printed was read, and what it said was published.

The real figures, recomputed from the artefacts now committed beside this test, are `p = 0.238`
and `0.0936`. The conclusion did not change; the strength of the evidence was stated more
precisely than the data supported.

So this file asserts the page against the run, in both directions:

- every number the page prints is recomputed here from `head-to-head/results/2026-09-10/`;
- and the artefacts themselves are checked for the things that must never be committed alongside
  them, because `.gitignore` had to be widened to admit this directory at all.

A fixture is a claim about the world. These are the world.
"""
import json
import re
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from senbonzakura.metrics import permutation_p  # noqa: E402

ARMS = ROOT / "head-to-head" / "results" / "2026-09-10"
PAGE = ROOT / "docs" / "guide" / "benchmark.md"
SEEDS = (42, 43, 44, 45, 46)


def _series(tool, kind, key):
    out = []
    for seed in SEEDS:
        doc = json.loads((ARMS / f"{kind}-{tool}-seed{seed}.json").read_text(encoding="utf-8"))
        out.append(doc[key])
    return out


@pytest.fixture(scope="module")
def drift():
    return _series("senbon", "drift", "kl"), _series("heretic", "drift", "kl")


# ── the artefacts are all there ──────────────────────────────────────────────────────

def test_every_arm_of_both_tools_is_present():
    """Ten arms, three instruments. A partial set published as a whole one is the failure the
    run's own manifest checking exists for, one level up.
    """
    missing = [f"{kind}-{tool}-seed{seed}.json"
               for kind in ("scored", "refusal", "drift")
               for tool in ("senbon", "heretic")
               for seed in SEEDS
               if not (ARMS / f"{kind}-{tool}-seed{seed}.json").is_file()]
    assert not missing, f"the published run is missing {len(missing)} artefact(s): {missing[:5]}"


# ── the page, against the run ────────────────────────────────────────────────────────

def test_the_published_p_value_recomputes(drift):
    """THE NUMBER THAT WAS WRONG. 0.405 was a property of a fixture; this is the run."""
    sen, her = drift
    got = permutation_p(sen, her, seed=0)
    assert round(got, 3) == 0.238, f"the page says p = 0.238 and the artefacts give {got:.4f}"
    assert f"p = {round(got, 3)}" in PAGE.read_text(encoding="utf-8") or "0.238" in PAGE.read_text(
        encoding="utf-8"), "the page and the artefacts disagree about p"


def test_the_published_means_and_medians_recompute(drift):
    sen, her = drift
    assert round(statistics.mean(sen), 4) == 0.0545
    assert round(statistics.mean(her), 4) == 0.1227
    assert round(statistics.median(sen), 4) == 0.0510
    assert round(statistics.median(her), 4) == 0.0573


def test_the_published_worst_seeds_recompute(drift):
    """The other number that was wrong: senbonzakura's worst seed, quoted as 0.0605."""
    sen, her = drift
    assert round(max(sen), 4) == 0.0936, "the page quotes senbonzakura's worst seed"
    assert round(max(her), 4) == 0.3591


def test_the_outlier_is_the_seed_the_page_names(drift):
    _sen, her = drift
    assert SEEDS[her.index(max(her))] == 43, "the page names seed 43 as the outlier"


def test_dropping_the_outlier_gives_the_figure_the_page_quotes(drift):
    sen, her = drift
    without = [v for v in her if v != max(her)]
    assert round(statistics.mean(without), 4) == 0.0636
    assert statistics.mean(without) > statistics.mean(sen), (
        "the page says Heretic's mean is still above ours without the outlier, narrowly")


def test_heretic_is_marginally_ahead_on_refusal_not_behind(drift):
    """The direction I reported backwards for a month, pinned so it cannot happen again."""
    sen = _series("senbon", "refusal", "refusal")
    her = _series("heretic", "refusal", "refusal")
    assert max(her) == 0.0, "every Heretic arm removed hard refusal completely"
    assert max(sen) > 0.0, "one of ours did not, which is what 'marginally ahead' means"
    assert statistics.mean(her) < statistics.mean(sen)


def test_the_page_does_not_still_carry_the_fixture_numbers():
    text = PAGE.read_text(encoding="utf-8")
    assert "0.405" not in text, "the fixture's p-value is back on the page"
    assert "worst seed | 0.0605" not in text


# ── what must never be committed beside them ─────────────────────────────────────────

def test_the_published_arms_carry_no_prompt_text():
    """`.gitignore` was widened to admit this directory, so the widening is checked here.

    Every string in every artefact is a label, a number, a token, or a filename. A long free-text
    string would mean a prompt or a completion had come along with the aggregate.
    """
    # The one long string these artefacts legitimately carry: `drift` records what measured the
    # number, which is the point of the field. Named rather than allowed for by raising the
    # length limit, so a NEW long string anywhere still fails.
    describes_the_instrument = {"instrument", "kl_ci_method", "precision_note", "budget_warning"}
    offenders = []

    def walk(node, where):
        if isinstance(node, str):
            if len(node) > 60 and where.rsplit(".", 1)[-1] not in describes_the_instrument:
                offenders.append((where, node[:80]))
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{where}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{where}[{i}]")

    for f in sorted(ARMS.glob("*.json")):
        walk(json.loads(f.read_text(encoding="utf-8")), f.name)
    assert not offenders, f"long free text in a published artefact: {offenders[:3]}"


def test_the_published_arms_carry_no_machine_paths():
    """The same defect found in the shipped wheel's own blob the same day."""
    bad = [f.name for f in ARMS.glob("*.json")
           if re.search(r'"(?:/home/|/tmp/|/root/|/Users/|[A-Za-z]:\\\\)', f.read_text(encoding="utf-8"))]
    assert not bad, f"these carry the machine they were measured on: {bad}"


def test_nothing_but_aggregate_json_and_prose_is_committed_here():
    """The exception admits `.json` and `.md`. Anything else means the widening slipped."""
    unexpected = [f.name for f in ARMS.iterdir()
                  if f.is_file() and f.suffix not in (".json", ".md")]
    assert not unexpected, f"unexpected file types in the published run: {unexpected}"
