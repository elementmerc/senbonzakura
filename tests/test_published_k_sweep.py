# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The project's most-quoted number, recomputed from data a reader can open.

WHAT WAS WRONG

`p = 0.016` is the multi-direction negative result, and it appears in the README, on the docs
index, in the CHANGELOG and on `docs/guide/what-we-know.md`. The ten per-seed drift values behind
it existed only in `private/results/2026-09-07-k-sweep-read.md`, and `private/` is git-excluded, so
nothing in the published repository let anyone reach the figure. A reviewer on 2026-09-10
reproduced it by recovering those numbers from a working note, which is not a route a reader has.

Two neighbouring pages were corrected for exactly this in the same cycle (`docs/comparison.md` now
says every figure comes from `tools/ci/leak_sweep.py`, with a test comparing the two, and
`docs/guide/benchmark.md` now admits no record of its run is kept). The page carrying the
project's most-quoted claim made no such admission.

These are ten floats and no prompts, so there was never a reason for them not to be committed.
`evidence/k-sweep-2026-08-13/drift-per-seed.json` holds them and this recomputes both published
figures from that file, through the project's own estimator.
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
    because it is what turns "twice the collateral damage" into "1.4 to 1.9 times".
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


def test_the_published_ratio_range_recomputes_and_names_its_estimator(arms):
    """1.4 to 1.9, on MEANS, which is what the pages now say.

    The range was published as "1.5 to 1.9" with no estimator named. On the means over five
    seeds the ratios are 1.88 with every seed and 1.45 dropping the outlying two-direction seed,
    so 1.5 arrives only through an unstated switch to medians, and rounding 1.45 up to 1.5
    widened this project's own negative result against itself.
    """
    k1, k2, _ = arms
    everything = statistics.mean(k2) / statistics.mean(k1)
    without_worst = statistics.mean([v for v in k2 if v != max(k2)]) / statistics.mean(k1)
    assert round(everything, 1) == 1.9, f"the pages say 1.9 and the data gives {everything:.3f}"
    assert round(without_worst, 1) == 1.4, f"the pages say 1.4 and the data gives {without_worst:.3f}"


def test_the_published_hard_refusal_column_recomputes(arms):
    """0.1% and 0.3%, the premise that makes the whole comparison readable.

    The argument on every page quoting this run is drift AT MATCHED REFUSAL, and until
    2026-09-25 that column traced to no artefact at all: the drift values were committed and
    the refusal values they are conditioned on were not. Recovered from the run's own database
    and recomputed here the same way the p-value is.
    """
    _, _, doc = arms
    hard = doc["hard_refusal"]
    assert len(hard["k1"]) == len(hard["k2"]) == len(doc["seeds"]) == 5
    assert statistics.mean(hard["k1"]) == pytest.approx(0.001, abs=5e-6)
    assert statistics.mean(hard["k2"]) == pytest.approx(0.003, abs=5e-6)
    assert "hard_refusal_note" in doc and "score" in doc["hard_refusal_note"]


# ── the caveat, on every surface that quotes the run ─────────────────────────────────
#
# The evidence file has carried its caveat since it was committed, and one documentation page
# carried it faithfully. The documents that travel furthest did not: the CHANGELOG said the
# comparison was finally run "properly" four paragraphs after telling the reader the old filter
# had been replaced, and `paper.md` reported the run as a non-reproduction of two papers whose
# directions are chosen deliberately. A caveat that lives only where a careful reader already
# is, is not a caveat.
#
# Same shape as `tests/test_published_head_to_head.py`: the artefact is the authority and the
# prose is checked against it, so neither can move without the other.

ROOT_MD = ROOT

#: Surfaces that quote the 2026-08-13 arms and must therefore carry the limit those arms have.
CAVEATED_SURFACES = (
    "CHANGELOG.md",
    "paper.md",
    "docs/guide/what-we-know.md",
    "docs/guide/prior-art.md",
    "docs/architecture.md",
    "man/senbonzakura.1",
    # THE FILE A SCEPTIC OPENS FIRST, added 2026-09-25 once it carried the caveat. It was the last
    # surface quoting these arms without one, which is the wrong way round: a reproduction document
    # is where somebody goes to check the claim rather than to read it.
    "REPRODUCING.md",
)

#: Ways a surface may legitimately word "the old filter accepted everything", collected from the
#: surfaces themselves rather than imposed, because a page is allowed its own register.
_FILTER_WORDING = re.compile(
    r"accept(?:s|ed)? (?:every|any|everything)|could not reject|unselective",
    re.IGNORECASE)

#: Files that quote the superseded "1.5 to 1.9" and are owned by a different change in this
#: cycle. This constant exists so the debt is named in the suite rather than remembered by
#: somebody, and it shrinks as each surface is corrected: `REPRODUCING.md` came off it on
#: 2026-09-25 and now states both means and the median it was confused with. The man page is
#: what is left.
#: EMPTY, AND IT HAS TO STAY THAT WAY. This existed for a few hours on 2026-09-25 while the ratio
#: correction reached one surface at a time, and an exemption list that outlives its migration is
#: how a corrected figure survives in the one place nobody re-reads. `man/senbonzakura.1` was the
#: last entry and installs to `share/man/man1`, so it reaches every user of a released wheel.
RATIO_NOT_YET_CORRECTED = ()


def _git_ignored():
    """Repository-relative paths git ignores, or an empty set when git cannot answer.

    Empty on failure by design: one CI job exports the tree without its history and runs the suite
    there, so `git` either is absent or has nothing to say. Falling back to scanning everything is
    the safe direction, because this guard exists to catch a claim that ships without its caveat.
    """
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(ROOT_MD), "ls-files", "--others", "--ignored",
                              "--exclude-standard", "-z"],
                             capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return set()
    if out.returncode != 0:
        return set()
    return {n for n in out.stdout.split("\0") if n}


def _public_docs():
    """Every public prose surface, minus the built VitePress output and private trees.

    The man page is in here because it installs to `share/man/man1` and reaches every user, and
    it has already gone stale once where nobody looks: see `tests/test_the_manual_that_ships.py`.
    """
    paths = [*ROOT_MD.rglob("*.md"), ROOT_MD / "man" / "senbonzakura.1"]
    ignored = _git_ignored()
    for path in sorted(p for p in paths if p.exists()):
        rel = path.relative_to(ROOT_MD).as_posix()
        # BUILD OUTPUT IS NOT DOCUMENTATION, and this list does not depend on git being present.
        # `dist-` with no slash is deliberate: it catches `dist-pypi/`, `dist-manylinux/`,
        # `dist-checker/` and any staging directory a release invents next, which is the shape
        # that broke this guard. A 2026-09-26 release staged a copy of the CHANGELOG under
        # `dist-release/` so the artefacts could be attached to the GitHub Release, and the suite
        # then failed on a developer machine and nowhere else.
        if rel.startswith(("private/", "docs/.vitepress/", "node_modules/", ".venv/",
                           "dist/", "dist-", "build/", "htmlcov/", ".tox/", "site/")):
            continue
        # ANYTHING GIT IGNORES CANNOT REACH A READER, so scanning it can only produce failures
        # that depend on what happens to be lying around. On 2026-09-26 a staging directory
        # holding a copy of the CHANGELOG, created to attach artefacts to the GitHub Release,
        # failed this guard on a developer machine and on nothing else.
        #
        # The prefix list above is kept rather than replaced: it covers `private/`, which IS
        # tracked in some checkouts, and it keeps working in the CI job that runs this suite from
        # an exported tree with no `.git` at all, where `_git_ignored` returns nothing.
        if rel in ignored:
            continue
        yield rel, path


@pytest.mark.parametrize("name", CAVEATED_SURFACES)
def test_every_surface_quoting_the_k_sweep_carries_its_caveat(name, arms):
    _, _, doc = arms
    assert "ARBITRARY" in doc["caveat"], "the evidence file has lost the caveat this pins"
    # WHITESPACE NORMALISED, because prose wraps and a regex with a literal space does not.
    # `REPRODUCING.md` carried the caveat correctly and failed this test on a line break
    # falling between "accept" and "every". A guard that depends on where an author
    # happened to wrap a sentence is measuring the formatting, not the claim.
    text = " ".join((ROOT_MD / name).read_text(encoding="utf-8").split())
    assert "arbitrary" in text.lower(), (
        f"{name} quotes the 2026-08-13 arms and never says the second directions were arbitrary")
    assert _FILTER_WORDING.search(text), (
        f"{name} calls the second directions arbitrary without saying why: the filter that chose "
        f"them accepted every candidate it was given, which is the whole reason the word applies")


def test_no_surface_quotes_the_run_without_being_on_the_caveat_list():
    """A NEW page quoting these figures must join the list above, not appear quietly beside it.

    The per-seed drift values are distinctive enough to be a fingerprint for this run. `0.016`
    on its own is not: `docs/guide/limits.md` carries an unrelated 0.016 and the writeups carry
    another, so matching on it alone would make this test cry wolf until somebody deleted it.
    """
    fingerprint = re.compile(r"0\.0497|0\.0932|p = 0\.016\b")
    stray = [rel for rel, path in _public_docs()
             if fingerprint.search(path.read_text(encoding="utf-8"))
             and rel not in CAVEATED_SURFACES]
    assert not stray, (
        f"these quote the 2026-08-13 arms and are not on the caveat list: {stray}")


def test_the_superseded_ratio_range_is_gone_from_the_surfaces_it_can_be_gone_from():
    """1.5 to 1.9 is the range with the estimator unstated and the low end rounded up."""
    stray = [rel for rel, path in _public_docs()
             if "1.5 to 1.9" in path.read_text(encoding="utf-8")
             and rel not in RATIO_NOT_YET_CORRECTED]
    assert not stray, (
        f"these still quote the superseded 1.5 to 1.9 range: {stray}. The means over five seeds "
        f"are 1.9 and 1.4; 1.5 is the median ratio, and mixing the two widened the project's own "
        f"negative result.")


def test_the_evidence_carries_no_prompts():
    """It is ten floats. If it ever grows prompt text, it stops being safe to commit."""
    raw = EVIDENCE.read_text(encoding="utf-8")
    doc = json.loads(raw)
    assert set(doc["drift_kl"]) == {"k1", "k2"}
    assert all(isinstance(v, float) for arm in doc["drift_kl"].values() for v in arm)
