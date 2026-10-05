# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The checker reads a model card, and refuses to audit what it cannot read.

WHY THIS ADAPTER NEEDS MORE HOSTILE TESTS THAN THE OTHERS

A card is unstructured prose, so this reader can be wrong in a way a JSON reader cannot, and being
wrong here means publishing a confident false audit of somebody else's work. Most of what follows
is therefore about what it must NOT claim.

THE FIXTURES' BASIS, AND ONE OF THEM IS A REAL FALSE POSITIVE

The card excerpts here are shortened from four cards fetched on 2026-10-06:
`199-biotechnologies/gemma-4-abliterated`, `empero-ai/Qwen3.8-4B-Distill`,
`wangzhang/gemma-4-31B-it-abliterated` and `huihui-ai/Huihui-Qwen3.5-27B-abliterated`.

The wangzhang line is the important one. Its card carries "| Optimization trials completed | 60/60
|", which the first version of this reader scored as a perfect result on a sample of 60 and would
have flagged. It is a progress counter. That card is also the strongest counter-example to the whole
check: it reports 7 refusals in 100 against a baseline of 99 in 100, shows three trials, and says
"We report 7/100 refusals honestly". A check that fired on it would have been attacking the one
publisher already doing the right thing, so the guard that stops it is tested here directly.
"""
import pytest
from senbonzakura_check import cardread
from senbonzakura_check.adapters import detect, normalise
from senbonzakura_check.adapters.model_card import ModelCardAdapter
from senbonzakura_check.loaders import MAX_CARD_BYTES, LoaderError, load_document

#: Shortened from the real 199 Biotechnologies card: a saturated suite and two absolute claims.
SATURATED_CARD = """# Gemma 4 31B Abliterated

**Zero intelligence loss** with zero capability degradation.

| Category | Baseline | Abliterated |
|---|---|---|
| Math (8 prompts) | 8/8 | 8/8 |
| Coding (8 prompts) | 8/8 | 8/8 |
| **Total** | **50/50 (100%)** | **50/50 (100%)** |

Generation is 16% faster than baseline.
"""

#: Shortened from the real wangzhang card: a counter that is not a score, beside honest scores.
CAREFUL_CARD = """# Gemma 4 31B abliterated

| Metric | Value |
|---|---|
| **Refusals (private eval dataset, 100 prompts)** | **7/100** |
| Baseline refusals (original model) | 99/100 |
| Optimization trials completed | 60/60 |

**We report 7/100 refusals honestly.** This is a measured number.
"""

#: Shortened from the real Empero card: benchmark numbers, no fractions, no absolute claim.
PLAIN_CARD = """---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
---
# Qwen3.8-4B-Distill

| Benchmark | Base | Ours |
|---|---|---|
| gsm8k_cot exact_match | 0.850 | 0.785 |
"""


def write_card(tmp_path, text, name="README.md"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ── the arithmetic, which is the whole claim ──────────────────────────────────────────


def test_the_interval_matches_the_main_package_exactly():
    """The copy must not drift. See `cardread.wilson_interval`'s docstring for why it is a copy."""
    import sys
    sys.path.insert(0, "src")
    from senbonzakura.metrics import wilson_interval as canonical

    for count, n in [(0, 1), (1, 1), (50, 50), (43, 50), (7, 100), (1121, 1319), (0, 30), (15, 30)]:
        mine = cardread.wilson_interval(count, n)
        theirs = canonical(count, n)
        assert mine == pytest.approx(theirs, abs=1e-12), f"drifted at {count}/{n}"


def test_an_empty_sample_is_the_widest_possible_interval():
    assert cardread.wilson_interval(0, 0) == (0.0, 1.0)


def test_the_boundary_for_a_perfect_fifty():
    """The published worked example: 43/50 is the lowest score that still overlaps 50/50.

    So a card claiming a perfect 50 out of 50 cannot distinguish a flawless model from one that
    lost seven answers in fifty, which is one in seven.
    """
    assert cardread.lowest_indistinguishable(50, 50) == 43
    lo_perfect, _ = cardread.wilson_interval(50, 50)
    _, hi_43 = cardread.wilson_interval(43, 50)
    _, hi_42 = cardread.wilson_interval(42, 50)
    assert hi_43 >= lo_perfect
    assert hi_42 < lo_perfect, "42/50 must separate, or the boundary is not 43"


def test_a_perfect_score_overlaps_something_lower_at_every_sample_size():
    """MEASURED, and it is why the check reads the WIDTH of the doubt rather than its existence.

    The obvious check, "does anything overlap a perfect score", is universal: it is true at n=5 and
    still true at n=100,000. Firing on it would flag a flawless hundred thousand, which is the
    cry-wolf this project refuses to ship. What varies, and what the check actually reads, is how
    far down the overlap reaches.
    """
    for n in (5, 50, 1000, 100000):
        assert cardread.lowest_indistinguishable(n, n) is not None


def test_the_doubt_narrows_as_the_sample_grows():
    """The numbers the check's threshold was chosen against."""
    gaps = {}
    for n in (50, 100, 200, 1000):
        lowest = cardread.lowest_indistinguishable(n, n)
        gaps[n] = (1.0 - lowest / n) * 100.0
    assert gaps[50] == pytest.approx(14.0)
    assert gaps[100] == pytest.approx(7.0)
    assert gaps[200] == pytest.approx(3.5)
    assert gaps[1000] == pytest.approx(0.7)
    assert gaps[50] > gaps[100] > gaps[200] > gaps[1000]


@pytest.mark.parametrize(("count", "n"), [(-1, 10), (11, 10), (5, 0), (5, -3)])
def test_an_impossible_pair_has_no_boundary(count, n):
    assert cardread.lowest_indistinguishable(count, n) is None


def test_a_tiny_sample_cannot_be_told_from_scoring_nothing_at_all():
    """At n=2 a perfect score overlaps even zero, so the walk runs to the bottom.

    Worth asserting rather than just covering: it is the extreme case of the whole finding. Two
    items proves nothing, and the arithmetic says so without needing a threshold.
    """
    assert cardread.lowest_indistinguishable(2, 2) == 0


# ── what counts as a claim, and what must not ─────────────────────────────────────────


def test_a_date_is_not_a_score():
    """`10/1/2025` must not be read as 10 out of 1, nor as anything else."""
    claims = cardread.extract_claims("Tested on 10/1/2025 and again on 3/12/2026.")
    assert claims == []


def test_a_path_is_not_a_score():
    assert cardread.extract_claims("See results/50/50 for the logs.") == []


def test_a_fraction_bigger_than_one_is_not_a_proportion():
    assert cardread.extract_claims("We ran 70/50 configurations.") == []


@pytest.mark.parametrize("total", [0, 1, cardread.MAX_TOTAL + 1])
def test_an_implausible_denominator_is_refused(total):
    assert cardread.extract_claims(f"scored 0/{total} on the suite") == []


def test_a_percentage_outside_the_range_is_not_a_rate():
    assert cardread.extract_claims("throughput rose 150%") == []
    assert [c["value"] for c in cardread.extract_claims("accuracy 94.5%")] == [0.945]


def test_a_counter_is_never_a_saturated_score():
    """The real false positive this guard exists for. See the module docstring."""
    claims = cardread.extract_claims("| Optimization trials completed | 60/60 |")
    assert len(claims) == 1
    assert claims[0]["count"] == 60
    assert claims[0]["is_count"] is True
    assert claims[0]["saturated"] is False, "a progress counter is not evidence of anything"


@pytest.mark.parametrize("line", [
    "| steps completed | 10/10 |",
    "| epochs | 3/3 |",
    "| shards written | 5/5 |",
    "| checkpoints saved | 2/2 |",
    "100% of tokens processed",
])
def test_counting_language_suppresses_saturation(line):
    assert all(not c["saturated"] for c in cardread.extract_claims(line))


def test_a_word_merely_containing_a_counting_word_does_not_suppress():
    """`profile` contains `file` and `overruns` contains `runs`. Word boundaries, not substrings."""
    assert cardread.looks_like_a_count("the profile scored 8/8") is False
    assert cardread.looks_like_a_count("files written 8/8") is True


def test_identical_figures_on_one_line_are_counted_once():
    """A total row holding the same figure twice is one claim, not two."""
    claims = cardread.extract_claims("| **Total** | **50/50 (100%)** | **50/50 (100%)** |")
    fractions = [c for c in claims if c["kind"] == "fraction"]
    assert len(fractions) == 1


def test_a_percentage_carries_no_denominator():
    """Inventing one to produce an interval is the fabrication this package exists to catch."""
    (claim,) = [c for c in cardread.extract_claims("scored 100% overall") if c["kind"] == "percent"]
    assert claim["count"] is None
    assert claim["total"] is None


# ── absolute claims ───────────────────────────────────────────────────────────────────


def test_the_absolute_claims_are_found_in_the_publishers_own_words():
    found = cardread.find_absolute_claims(SATURATED_CARD)
    assert "zero intelligence loss" in found
    assert "zero capability degradation" in found


def test_a_card_making_no_absolute_claim_reports_none():
    assert cardread.find_absolute_claims(CAREFUL_CARD) == []
    assert cardread.find_absolute_claims(PLAIN_CARD) == []


def test_an_absolute_claim_with_no_number_at_all_is_flagged():
    card = cardread.read_card("# M\n\nThis edit is **lossless**.\n")
    assert card["absolute_claim_without_any_number"] is True


def test_an_absolute_claim_beside_numbers_is_not_that_finding():
    """A weak measurement is a different finding from an absent one."""
    card = cardread.read_card(SATURATED_CARD)
    assert card["absolute_claims"]
    assert card["absolute_claim_without_any_number"] is False


# ── frontmatter ───────────────────────────────────────────────────────────────────────


def test_frontmatter_is_read_without_a_yaml_parser():
    fm = cardread.read_card(PLAIN_CARD)["frontmatter"]
    assert fm["license"] == "apache-2.0"
    assert fm["base_model"] == "Qwen/Qwen3.5-4B"


@pytest.mark.parametrize("text", [
    "# No frontmatter\n",
    "---\nunterminated: yes\n",
    "---\n---\n# empty frontmatter\n",
])
def test_frontmatter_absent_or_broken_yields_nothing_rather_than_a_guess(text):
    assert cardread.read_card(text)["frontmatter"] == {}


def test_a_nested_frontmatter_value_is_skipped_rather_than_half_read():
    text = "---\nlicense: mit\nwidget:\n  - text: hello\n---\n# M\n"
    fm = cardread.read_card(text)["frontmatter"]
    assert fm == {"license": "mit"}


def test_a_frontmatter_line_with_no_colon_is_ignored():
    text = "---\nlicense: mit\njust a line\n---\n# M\n"
    assert cardread.read_card(text)["frontmatter"] == {"license": "mit"}


# ── the adapter ───────────────────────────────────────────────────────────────────────


def test_the_adapter_owns_a_card_document(tmp_path):
    assert detect(load_document(write_card(tmp_path, SATURATED_CARD))) is ModelCardAdapter


def test_the_adapter_declines_anything_else():
    assert ModelCardAdapter.detects({"results": {}, "configs": {}, "versions": {}}) is False
    assert ModelCardAdapter.detects({}) is False


def test_the_saturated_card_normalises_with_the_boundary(tmp_path):
    n = normalise(load_document(write_card(tmp_path, SATURATED_CARD)))
    assert n["harness"] == "model card"
    assert n["artefact_kind"] == "model-card"
    assert n["saturated_total"] == 50
    assert n["lowest_indistinguishable_count"] == 43
    assert n["lowest_indistinguishable_share"] == pytest.approx(0.86)
    assert n["indistinguishable_gap_pp"] == pytest.approx(14.0)
    assert n["saturation"]["claim"]["gap_pp"] == pytest.approx(14.0)
    assert "Total" in n["saturated_context"]


def test_the_careful_card_does_not_look_saturated(tmp_path):
    """The publisher already doing the right thing must not be told their evidence is thin."""
    n = normalise(load_document(write_card(tmp_path, CAREFUL_CARD)))
    assert n["saturated_total"] is None
    assert n["saturation"] == {}, "no saturated claim means nothing for the check to read"


def test_a_fraction_becomes_a_metric_with_its_denominator(tmp_path):
    n = normalise(load_document(write_card(tmp_path, SATURATED_CARD)))
    fractions = [m for m in n["metrics"].values() if m["units"] == "proportion"]
    assert fractions, "a fraction supplies both halves, so it has units and an n"
    assert all(m["n"] for m in fractions)
    assert all(m["stderr"] is None for m in fractions)


def test_a_percentage_becomes_a_metric_with_no_denominator_and_no_units(tmp_path):
    """`16% faster` is a relative change, not a rate, so claiming units would misread it."""
    n = normalise(load_document(write_card(tmp_path, SATURATED_CARD)))
    percents = [m for m in n["metrics"].values() if m["units"] is None]
    assert percents
    assert all(m["n"] is None for m in percents)


def test_the_declared_licence_and_base_model_are_lifted(tmp_path):
    n = normalise(load_document(write_card(tmp_path, PLAIN_CARD)))
    assert n["declared_licence"] == "apache-2.0"
    assert n["declared_base_model"] == "Qwen/Qwen3.5-4B"


def test_a_card_with_no_claims_reports_none(tmp_path):
    n = normalise(load_document(write_card(tmp_path, "# Just a title\n\nSome prose.\n")))
    assert n["claim_count"] == 0
    assert n["metrics"] == {}
    assert n["eval_split"] is None and n["limit"] is None


# ── refusing a file that is not a card ────────────────────────────────────────────────


def test_an_empty_card_is_unchecked_rather_than_clean(tmp_path):
    with pytest.raises(LoaderError, match="no claim in it to check"):
        load_document(write_card(tmp_path, "   \n\n  \n"))


def test_a_card_that_is_not_text_is_refused(tmp_path):
    p = tmp_path / "README.md"
    p.write_bytes(b"\xff\xfe\x00 not utf-8 \xff")
    with pytest.raises(LoaderError, match="not text this tool can decode"):
        load_document(p)


def test_an_unreadable_card_is_refused(tmp_path):
    with pytest.raises(LoaderError, match="could not read it"):
        load_document(tmp_path / "absent.md")


def test_a_directory_named_like_a_card_is_refused(tmp_path):
    """`stat` succeeds on a directory and the read then fails, which is a different branch."""
    d = tmp_path / "README.md"
    d.mkdir()
    with pytest.raises(LoaderError, match="could not read it"):
        load_document(d)


def test_an_over_large_card_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("senbonzakura_check.loaders.MAX_CARD_BYTES", 64)
    with pytest.raises(LoaderError, match="a model card is prose"):
        load_document(write_card(tmp_path, "x" * 200))


def test_the_card_cap_is_generous_enough_for_a_real_card():
    """The real cards read were 2.4 KB to 10.5 KB. A cap near those would refuse the job."""
    assert MAX_CARD_BYTES > 1_000_000


def test_the_markdown_long_suffix_is_read_too(tmp_path):
    n = normalise(load_document(write_card(tmp_path, SATURATED_CARD, name="CARD.markdown")))
    assert n["artefact_kind"] == "model-card"


# ── end to end, through the checks ────────────────────────────────────────────────────


def test_the_saturated_check_fires_end_to_end(tmp_path):
    from senbonzakura_check.cli import inspect_file
    from senbonzakura_check.registry import load_checks

    findings, _skipped, problem = inspect_file(
        write_card(tmp_path, SATURATED_CARD), load_checks())
    assert problem is None
    assert "a-perfect-score-on-a-sample-too-small-to-show-it" in [f.check_id for f in findings]


def test_the_saturated_check_stays_quiet_on_the_careful_card(tmp_path):
    from senbonzakura_check.cli import inspect_file
    from senbonzakura_check.registry import load_checks

    findings, _skipped, problem = inspect_file(
        write_card(tmp_path, CAREFUL_CARD), load_checks())
    assert problem is None
    assert "a-perfect-score-on-a-sample-too-small-to-show-it" not in [
        f.check_id for f in findings]


def test_the_unevidenced_absolute_check_fires_end_to_end(tmp_path):
    from senbonzakura_check.cli import inspect_file
    from senbonzakura_check.registry import load_checks

    card = "# M\n\nThis release has **no capability loss** whatsoever.\n"
    findings, _skipped, problem = inspect_file(write_card(tmp_path, card), load_checks())
    assert problem is None
    assert "an-absolute-claim-with-no-number-anywhere-on-the-card" in [
        f.check_id for f in findings]


def test_a_card_with_numbers_does_not_fire_the_unevidenced_check(tmp_path):
    from senbonzakura_check.cli import inspect_file
    from senbonzakura_check.registry import load_checks

    findings, _skipped, problem = inspect_file(
        write_card(tmp_path, SATURATED_CARD), load_checks())
    assert "an-absolute-claim-with-no-number-anywhere-on-the-card" not in [
        f.check_id for f in findings]
