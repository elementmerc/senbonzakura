# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Many results read as one family, and the refusals that keep the reading honest.

THE PROPERTY THESE TESTS EXIST TO PROTECT

The aggregator must never average point estimates. The whole module is built around one question,
"is a single value consistent with every member", and the tests that matter most are the ones
asserting it says no: a family containing one member that is genuinely different has to come back
as not one thing, and a pooled figure must not appear for it.

The second property is that nothing is dropped quietly. A member with no interval, an artefact with
no such metric, and a mix of interval sources all have to reach the reader, because every one of
them makes the family look more consistent than the evidence if it is swallowed.
"""
from __future__ import annotations

import math
import pathlib

import pytest
from senbonzakura_check import family


def doc(model, value, n=100, *, metric="refusal", estimator="senbonzakura-ruler",
        units="proportion", interval=None, key=None):
    block = {"metric": metric, "value": value, "estimator": estimator, "units": units, "n": n}
    if interval is not None:
        block["interval"] = interval
    return {"model": model, "metrics": {key or metric: block}}


def docs(*rows):
    return [(f"{m}.json", d) for m, d in rows]


def four_similar():
    return docs(("a", doc("a", 0.10)), ("b", doc("b", 0.12)), ("c", doc("c", 0.09)),
                ("d", doc("d", 0.11)))


# --- the verdict, which is the product ------------------------------------------------------


def test_a_family_whose_members_all_overlap_is_reported_as_one_thing():
    res = family.aggregate(four_similar(), metric="refusal")
    assert res.one_figure_defensible is True
    assert res.separated == ()
    assert res.common_low is not None and res.common_low <= res.common_high
    assert "Nothing here distinguishes the members" in family.verdict_sentence(res)


def test_one_member_that_is_genuinely_different_breaks_the_family():
    rows = docs(("a", doc("a", 0.10)), ("b", doc("b", 0.12)), ("c", doc("c", 0.09)),
                ("odd", doc("odd", 0.40)))
    res = family.aggregate(rows, metric="refusal")
    assert res.one_figure_defensible is False
    assert res.common_low is None and res.common_high is None
    # The separated pairs name the member that is different, which is the actionable output.
    assert res.separated, "a 40% member against three around 10% must be separated"
    assert all("odd" in (lo.model, hi.model) for lo, hi, _ in res.separated)
    assert "is not one thing" in family.verdict_sentence(res)


def test_no_pooled_figure_is_offered_for_a_family_the_evidence_separates():
    rows = docs(("a", doc("a", 0.10)), ("b", doc("b", 0.12)), ("c", doc("c", 0.09)),
                ("odd", doc("odd", 0.40)))
    res = family.aggregate(rows, metric="refusal")
    # THE DEFECT THE MODULE EXISTS TO AVOID. The mean of these four is 17.75%, which describes no
    # member. Nothing in the result may offer a single number for them.
    assert res.pooled is None


def test_a_pooled_figure_is_offered_when_it_cannot_mislead():
    res = family.aggregate(four_similar(), metric="refusal")
    assert res.pooled is not None
    assert res.pooled["n"] == 400
    assert res.pooled["count"] == 10 + 12 + 9 + 11
    assert res.pooled["low"] < res.pooled["value"] < res.pooled["high"]
    # Pooling 400 prompts has to be tighter than any member's 100, or the pooling bought nothing.
    widest = max(m.high - m.low for m in res.members)
    assert res.pooled["high"] - res.pooled["low"] < widest
    assert "not about any one model" in res.pooled["means"]


def test_the_sentence_says_so_when_no_member_carries_an_interval():
    rows = docs(("a", doc("a", 0.1, n=None, units="divergence")),
                ("b", doc("b", 0.2, n=None, units="divergence")),
                ("c", doc("c", 0.3, n=None, units="divergence")))
    res = family.aggregate(rows, metric="refusal")
    assert res.one_figure_defensible is None
    assert res.pooled is None
    assert "Nothing can be said" in family.verdict_sentence(res)


# --- nothing is dropped quietly -------------------------------------------------------------


def test_a_member_with_no_interval_is_named_rather_than_skipped():
    rows = docs(("a", doc("a", 0.10)), ("b", doc("b", 0.12)), ("c", doc("c", 0.09)),
                ("bare", doc("bare", 0.11, n=None)))
    res = family.aggregate(rows, metric="refusal")
    assert [m.model for m in res.without_interval] == ["bare"]
    assert any("bare" in note and "no interval" in note for note in res.notes)


def test_an_artefact_without_the_metric_is_named_rather_than_counted_out():
    rows = [*four_similar(), ("other.json", doc("e", 0.5, metric="kl"))]
    res = family.aggregate(rows, metric="refusal")
    assert len(res.members) == 4
    assert any("other.json" in note for note in res.notes)


def test_mixed_interval_sources_are_flagged():
    rows = docs(("a", doc("a", 0.10)), ("b", doc("b", 0.12)),
                ("c", doc("c", 0.11, interval=[0.05, 0.19])))
    res = family.aggregate(rows, metric="refusal")
    bases = {m.interval_basis for m in res.members}
    assert bases == {"wilson", "artefact"}
    assert any("more than one source" in note for note in res.notes)


def test_the_artefacts_own_interval_wins_over_a_recomputed_one():
    rows = docs(("a", doc("a", 0.10, interval=[0.02, 0.30])), ("b", doc("b", 0.12)),
                ("c", doc("c", 0.09)))
    res = family.aggregate(rows, metric="refusal")
    a = next(m for m in res.members if m.model == "a")
    assert (a.low, a.high) == (0.02, 0.30)
    assert a.interval_basis == "artefact"


def test_a_malformed_artefact_interval_falls_back_to_the_computed_one():
    for bad in ([0.3, 0.1], ["x", 0.2], [0.1], [0.1, 0.2, 0.3], "nope", {"low": 1}, [True, 0.2]):
        rows = docs(("a", doc("a", 0.10, interval=bad)), ("b", doc("b", 0.12)),
                    ("c", doc("c", 0.09)))
        res = family.aggregate(rows, metric="refusal")
        a = next(m for m in res.members if m.model == "a")
        assert a.interval_basis == "wilson", bad


def test_no_pooled_figure_without_a_sample_size_on_every_member():
    rows = docs(("a", doc("a", 0.10)), ("b", doc("b", 0.12)),
                ("c", doc("c", 0.11, interval=[0.05, 0.19], n=None)))
    res = family.aggregate(rows, metric="refusal")
    assert res.one_figure_defensible is True
    assert res.pooled is None
    assert any("which artefacts happened to record an n" in note for note in res.notes)


def test_a_non_proportion_metric_gets_no_pooled_figure_even_when_consistent():
    rows = docs(("a", doc("a", 0.01, units="divergence", interval=[0.0, 0.05])),
                ("b", doc("b", 0.02, units="divergence", interval=[0.0, 0.06])),
                ("c", doc("c", 0.03, units="divergence", interval=[0.0, 0.07])))
    res = family.aggregate(rows, metric="refusal")
    assert res.one_figure_defensible is True
    assert res.pooled is None


# --- the refusals ---------------------------------------------------------------------------


def test_two_artefacts_is_a_comparison_and_the_refusal_says_which_mode_to_use():
    rows = docs(("a", doc("a", 0.10)), ("b", doc("b", 0.12)))
    with pytest.raises(family.FamilyError, match="--pair"):
        family.aggregate(rows, metric="refusal")


def test_mixing_estimators_is_refused_rather_than_reconciled():
    rows = docs(("a", doc("a", 0.10)), ("b", doc("b", 0.12)),
                ("c", doc("c", 0.09, estimator="heretic-keyword")))
    with pytest.raises(family.FamilyError, match="different quantities"):
        family.aggregate(rows, metric="refusal")


def test_mixing_units_is_refused():
    rows = docs(("a", doc("a", 0.10)), ("b", doc("b", 0.12)),
                ("c", doc("c", 0.09, units="percent")))
    with pytest.raises(family.FamilyError, match="units"):
        family.aggregate(rows, metric="refusal")


def test_two_readings_of_one_model_are_refused_and_named():
    rows = [("first.json", doc("a", 0.10)), ("second.json", doc("a", 0.30)),
            ("third.json", doc("b", 0.12)), ("fourth.json", doc("c", 0.09))]
    with pytest.raises(family.FamilyError, match=r"first\.json and second\.json"):
        family.aggregate(rows, metric="refusal")


def test_a_value_that_is_not_a_number_is_refused_rather_than_skipped():
    for bad in (None, "0.1", True, [0.1]):
        rows = docs(("a", doc("a", bad)), ("b", doc("b", 0.12)), ("c", doc("c", 0.09)),
                    ("d", doc("d", 0.11)))
        with pytest.raises(family.FamilyError, match="not a number"):
            family.aggregate(rows, metric="refusal")


def test_a_metric_absent_from_every_artefact_says_so_as_a_spelling_problem():
    with pytest.raises(family.FamilyError, match="spelling"):
        family.aggregate(four_similar(), metric="refussal")


def test_a_size_for_a_model_nobody_supplied_is_refused():
    with pytest.raises(family.FamilyError, match="typo in the model name"):
        family.aggregate(four_similar(), metric="refusal", sizes={"nonexistent": 7})


def test_a_metric_block_that_is_not_a_mapping_is_ignored_without_crashing():
    rows = four_similar()
    rows[0][1]["metrics"]["junk"] = "not a block"
    res = family.aggregate(rows, metric="refusal")
    assert len(res.members) == 4


def test_a_document_with_no_metrics_block_at_all_is_treated_as_missing_the_metric():
    rows = [*four_similar(), ("empty.json", {"model": "e"})]
    res = family.aggregate(rows, metric="refusal")
    assert len(res.members) == 4
    assert any("empty.json" in note for note in res.notes)


def test_a_prefixed_metric_key_is_matched_on_the_declared_metric_field():
    rows = docs(("a", doc("a", 0.10, key="arm_a.refusal")),
                ("b", doc("b", 0.12, key="arm_b.refusal")),
                ("c", doc("c", 0.09, key="refusal")))
    res = family.aggregate(rows, metric="refusal")
    assert len(res.members) == 3


def test_a_model_name_falls_back_to_the_source_when_the_artefact_has_none():
    rows = [("a.json", {"metrics": {"refusal": {"metric": "refusal", "value": 0.1,
                                                "estimator": "r", "units": "proportion", "n": 100}}}),
            ("b.json", doc("b", 0.12, estimator="r")), ("c.json", doc("c", 0.09, estimator="r"))]
    res = family.aggregate(rows, metric="refusal")
    assert "a.json" in [m.model for m in res.members]


# --- the trend ------------------------------------------------------------------------------


def test_no_trend_is_read_without_sizes():
    res = family.aggregate(four_similar(), metric="refusal", want_trend=True)
    assert res.trend["verdict"] == "no trend can be read"
    assert "supplied by the caller" in res.trend["why"]


def test_no_trend_is_read_when_every_pair_of_sizes_overlaps():
    sizes = {"a": 1e9, "b": 8e9, "c": 27e9, "d": 70e9}
    res = family.aggregate(four_similar(), metric="refusal", sizes=sizes, want_trend=True)
    assert res.trend["verdict"] == "no trend can be read"
    assert "does not order them" in res.trend["why"]


def test_a_consistent_direction_is_reported_without_claiming_a_slope():
    rows = docs(("a", doc("a", 0.02)), ("b", doc("b", 0.03)), ("c", doc("c", 0.40)),
                ("d", doc("d", 0.45)))
    sizes = {"a": 1e9, "b": 2e9, "c": 27e9, "d": 70e9}
    res = family.aggregate(rows, metric="refusal", sizes=sizes, want_trend=True)
    assert res.trend["verdict"] == "rises with size"
    assert res.trend["falling"] == 0
    assert "not a fitted slope" in res.trend["why"]


def test_the_opposite_direction_is_reported_too():
    rows = docs(("a", doc("a", 0.45)), ("b", doc("b", 0.40)), ("c", doc("c", 0.03)),
                ("d", doc("d", 0.02)))
    sizes = {"a": 1e9, "b": 2e9, "c": 27e9, "d": 70e9}
    res = family.aggregate(rows, metric="refusal", sizes=sizes, want_trend=True)
    assert res.trend["verdict"] == "falls with size"
    assert res.trend["rising"] == 0


def test_an_inconsistent_direction_refuses_to_pick_one():
    rows = docs(("a", doc("a", 0.02)), ("b", doc("b", 0.50)), ("c", doc("c", 0.03)),
                ("d", doc("d", 0.55)), ("e", doc("e", 0.04)))
    sizes = {"a": 1e9, "b": 2e9, "c": 4e9, "d": 8e9, "e": 16e9}
    res = family.aggregate(rows, metric="refusal", sizes=sizes, want_trend=True)
    assert res.trend["verdict"] == "the direction is inconsistent"
    assert res.trend["rising"] and res.trend["falling"]
    assert "describe the sample and not the lineage" in res.trend["why"]


def test_two_members_at_the_same_size_contribute_no_pair():
    rows = docs(("a", doc("a", 0.02)), ("b", doc("b", 0.50)), ("c", doc("c", 0.55)))
    # a is separated from both, but b and c share a size so their pair is skipped.
    sizes = {"a": 1e9, "b": 8e9, "c": 8e9}
    res = family.aggregate(rows, metric="refusal", sizes=sizes, want_trend=True)
    assert res.trend["rising"] == 2
    assert res.trend["falling"] == 0


def test_a_trend_needs_enough_sized_members_with_intervals():
    rows = docs(("a", doc("a", 0.02)), ("b", doc("b", 0.50)),
                ("c", doc("c", 0.55, n=None)))
    sizes = {"a": 1e9, "b": 8e9, "c": 27e9}
    res = family.aggregate(rows, metric="refusal", sizes=sizes, want_trend=True)
    assert res.trend["verdict"] == "no trend can be read"
    assert "both a size and an interval" in res.trend["why"]


def test_no_trend_block_at_all_unless_it_was_asked_for():
    assert family.aggregate(four_similar(), metric="refusal").trend is None


# --- the arithmetic agrees with the main package -------------------------------------------


def test_a_wilson_interval_computed_here_matches_the_count_it_came_from():
    rows = docs(("a", doc("a", 0.07)), ("b", doc("b", 0.12)), ("c", doc("c", 0.09)))
    res = family.aggregate(rows, metric="refusal")
    a = next(m for m in res.members if m.model == "a")
    from senbonzakura_check.cardread import wilson_interval
    assert (a.low, a.high) == wilson_interval(7, 100)
    assert math.isclose(a.low, 0.03431882, abs_tol=1e-7)


# --- through the command line ---------------------------------------------------------------


def write(tmp_path, name, value, n=200, *, estimator="senbonzakura-ruler", units="proportion"):
    import json as _json
    p = tmp_path / f"{name}.json"
    p.write_text(_json.dumps({
        "schema": "senbonzakura-score/1", "model": name, "label": f"{name}-run",
        "metrics": {"refusal": {"metric": "refusal", "value": value, "estimator": estimator,
                                "units": units, "n": n, "higher_is_better": False}}}))
    return str(p)


def run(argv):
    import io

    from senbonzakura_check import cli
    out = io.StringIO()
    status = cli.main(argv, out=out)
    return status, out.getvalue()


def a_separated_family(tmp_path):
    return [write(tmp_path, "Fam-1B", 0.02), write(tmp_path, "Fam-8B", 0.03),
            write(tmp_path, "Fam-27B", 0.41), write(tmp_path, "Fam-70B", 0.45)]


def a_consistent_family(tmp_path):
    return [write(tmp_path, "Fam-1B", 0.10), write(tmp_path, "Fam-8B", 0.11),
            write(tmp_path, "Fam-27B", 0.12)]


def test_the_cli_reports_a_separated_family_and_names_the_pairs(tmp_path):
    status, text = run([*a_separated_family(tmp_path), "--family", "refusal"])
    assert status == 0
    assert "is not one thing" in text
    assert "Differences the evidence supports" in text
    assert "Fam-1B below Fam-70B" in text
    # The two pairs that overlap must not be listed; their absence is the finding.
    assert "Fam-1B below Fam-8B" not in text
    assert "Pooled" not in text


def test_the_cli_reports_a_consistent_family_with_a_pooled_figure(tmp_path):
    status, text = run([*a_consistent_family(tmp_path), "--family", "refusal"])
    assert status == 0
    assert "Nothing here distinguishes the members" in text
    assert "Pooled:" in text
    assert "Differences the evidence supports" not in text


def test_the_cli_reports_a_trend_and_its_notes(tmp_path):
    paths = a_separated_family(tmp_path)
    status, text = run([*paths, "--family", "refusal", "--trend",
                        "--size", "Fam-1B=1B", "--size", "Fam-8B=8B",
                        "--size", "Fam-27B=27B", "--size", "Fam-70B=70B"])
    assert status == 0
    assert "Size trend: rises with size" in text


def test_the_cli_prints_a_note_when_an_artefact_lacks_the_metric(tmp_path):
    other = write(tmp_path, "Elsewhere", 0.5)
    import json as _json
    path = pathlib.Path(other)
    doc = _json.loads(path.read_text())
    doc["metrics"] = {"kl": {"metric": "kl", "value": 0.01, "estimator": "e", "units": "divergence",
                             "n": None}}
    path.write_text(_json.dumps(doc))
    status, text = run([*a_consistent_family(tmp_path), other, "--family", "refusal"])
    assert status == 0
    assert "carry no 'refusal' metric" in text


def test_the_cli_emits_json_carrying_the_verdict(tmp_path):
    import json as _json
    status, text = run([*a_separated_family(tmp_path), "--family", "refusal", "--trend",
                        "--size", "Fam-1B=1B", "--size", "Fam-8B=8B",
                        "--size", "Fam-27B=27B", "--size", "Fam-70B=70B", "--json"])
    assert status == 0
    doc = _json.loads(text)
    assert doc["one_figure_defensible"] is False
    assert doc["common_range"] is None
    assert len(doc["members"]) == 4
    assert doc["separated"] and doc["trend"]["verdict"] == "rises with size"
    assert all(m["interval"] for m in doc["members"])


def test_the_cli_json_carries_a_common_range_when_there_is_one(tmp_path):
    import json as _json
    status, text = run([*a_consistent_family(tmp_path), "--family", "refusal", "--json"])
    doc = _json.loads(text)
    assert doc["one_figure_defensible"] is True
    assert doc["common_range"][0] <= doc["common_range"][1]
    assert doc["pooled"]["n"] == 600


def test_the_cli_refuses_family_and_pair_together(tmp_path):
    status, text = run([*a_consistent_family(tmp_path)[:2], "--family", "refusal", "--pair"])
    assert status == 2
    assert "two different questions" in text


def test_the_cli_is_fatal_on_an_unreadable_member(tmp_path):
    bad = tmp_path / "broken.json"
    bad.write_text("{not json")
    status, text = run([*a_consistent_family(tmp_path), str(bad), "--family", "refusal"])
    assert status == 2
    # FATAL RATHER THAN SKIPPED. An aggregate over the readable subset describes a family the
    # caller did not name, and nothing in the output would say so.
    assert "Nothing was aggregated" in text


def test_the_cli_passes_a_family_error_through(tmp_path):
    paths = [write(tmp_path, "a", 0.10), write(tmp_path, "b", 0.11),
             write(tmp_path, "c", 0.12, estimator="heretic-keyword")]
    status, text = run([*paths, "--family", "refusal"])
    assert status == 2
    assert "different quantities" in text


@pytest.mark.parametrize(("bad", "expect"), [
    ("Fam-1B", "not MODEL=PARAMS"),
    ("=8B", "not MODEL=PARAMS"),
    ("Fam-1B=", "not MODEL=PARAMS"),
    ("Fam-1B=big", "where a number belongs"),
    ("Fam-1B=0", "positive parameter count"),
    ("Fam-1B=-4B", "positive parameter count"),
])
def test_every_malformed_size_is_refused_with_its_own_sentence(tmp_path, bad, expect):
    status, text = run([*a_consistent_family(tmp_path), "--family", "refusal", "--size", bad])
    assert status == 2
    assert expect in text


def test_a_size_named_twice_is_refused(tmp_path):
    status, text = run([*a_consistent_family(tmp_path), "--family", "refusal",
                        "--size", "Fam-1B=1B", "--size", "Fam-1B=2B"])
    assert status == 2
    assert "twice" in text


def test_a_size_suffix_is_read_and_a_bare_number_is_too(tmp_path):
    from senbonzakura_check.cli import _parse_sizes
    sizes, problem = _parse_sizes(["a=1k", "b=2M", "c=3B", "d=4T", "e=5"])
    assert problem is None
    assert sizes == {"a": 1e3, "b": 2e6, "c": 3e9, "d": 4e12, "e": 5.0}


def test_the_table_marks_a_member_with_no_interval(tmp_path):
    bare = write(tmp_path, "Bare", 0.11, n=None)
    status, text = run([*a_consistent_family(tmp_path), bare, "--family", "refusal"])
    assert status == 0
    assert "no interval" in text
    assert "n unstated" in text


# --- several rows of one ranking are several members ----------------------------------------


BOARD = ("author/model_name,UGI,W/10\n"
         "Pub/Model-8B,32.1,7.2\n"
         "Pub/Model-27B,35.4,8.1\n"
         "Pub/Model-70B,38.9,8.6\n"
         "Other/Thing,20.0,5.0\n")


def test_a_published_ranking_yields_a_family_and_says_nothing_can_be_concluded(tmp_path):
    board = tmp_path / "board.csv"
    board.write_text(BOARD)
    status, text = run([str(board), "--family", "UGI", "--row", "Pub/Model-8B",
                        "--row", "Pub/Model-27B", "--row", "Pub/Model-70B"])
    assert status == 0
    # THE VALUABLE OUTPUT. A ranking publishes a score and no sample size, so a lineage listed on
    # one cannot be assessed as a lineage, and saying that is the finding rather than a shortfall.
    assert "Nothing can be said about UGI" in text
    assert "no interval" in text
    assert "Other/Thing" not in text


def test_several_rows_with_several_files_is_refused(tmp_path):
    board = tmp_path / "board.csv"
    board.write_text(BOARD)
    status, text = run([str(board), *a_consistent_family(tmp_path), "--family", "UGI",
                        "--row", "Pub/Model-8B", "--row", "Pub/Model-27B"])
    assert status == 2
    assert "members of ONE ranking" in text


def test_a_named_row_that_matches_nothing_is_fatal_in_family_mode(tmp_path):
    board = tmp_path / "board.csv"
    board.write_text(BOARD)
    status, text = run([str(board), "--family", "UGI", "--row", "Pub/Model-8B",
                        "--row", "Pub/Model-27B", "--row", "Pub/Nope"])
    assert status == 2
    assert "Pub/Nope" in text
    assert "Nothing was aggregated" in text


def test_several_rows_outside_family_mode_is_refused(tmp_path):
    board = tmp_path / "board.csv"
    board.write_text(BOARD)
    status, text = run([str(board), "--row", "Pub/Model-8B", "--row", "Pub/Model-27B"])
    assert status == 2
    assert "exactly one row is checked" in text


def test_one_row_outside_family_mode_still_works(tmp_path):
    board = tmp_path / "board.csv"
    board.write_text(BOARD)
    status, _text = run([str(board), "--row", "Pub/Model-8B", "--quiet"])
    assert status in (0, 1)
