# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The checker reads a row of a public ranking, and refuses to invent what the ranking omits.

THE PROPERTY THESE TESTS EXIST TO PROTECT

A public ranking of this kind publishes a score and nothing behind it. The adapter's job is to say
so truthfully, and the failure mode worth guarding is not a crash: it is the adapter helpfully
filling in a sample size from an assumption so that an interval can be computed. A fabricated
interval reads to every downstream reader as a figure that has been checked, which is worse than no
figure at all, so `n` and `stderr` staying None are asserted directly rather than left implied.

THE FIXTURE'S BASIS

The column names and the shape come from the UGI Leaderboard's own published data file, read on
2026-10-05: 1,326 rows and 71 columns, with no interval, standard error or sample size among them.
The values used here are the real published figures for two rows of it, which are quoted in
`private/outreach/2026-10-05-per-company-findings.md`. The file itself is 667 KB and is not
committed; these tests build the rows they need and download nothing.
"""
import pytest
from senbonzakura_check.adapters import detect, normalise
from senbonzakura_check.adapters.leaderboard_row import (
    LeaderboardRowAdapter,
    _as_number,
    quoted_precision,
)
from senbonzakura_check.loaders import MAX_ROWS, LoaderError, load_document

#: The columns this adapter meets in the wild, trimmed to the ones that exercise every branch:
#: an identity column, a known attribute, a flag that must not become a metric, scores quoted at
#: two different precisions, and a cell holding a dash.
HEADER = ("author/model_name,Model Link,Test Date,Is Thinking Model,"
          "UGI,W/10,NatInt,Avg Thinking Chars,Writing")

#: The real published figures for Latitude's Harbinger 24B and a third party's abliteration of it.
HARBINGER = "LatitudeGames/Harbinger-24B,https://hf.co/x,10/1/2025,False,36.5,6.8,27.93,0,5.1"
ABLITERATED = ("vprilepskii/Harbinger-24B-biprojected-norm-preserving-abliterated,"
               "https://hf.co/y,12/4/2025,False,43.6,7.8,26.25,0,-")


def write_csv(tmp_path, *rows, name="board.csv", header=HEADER):
    p = tmp_path / name
    p.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return p


# ── reading a cell ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("cell", "expected"), [
    ("36.5", 36.5), ("27.93", 27.93), ("0", 0.0), ("-12.5", -12.5),
    ("1,319", 1319.0), ("  7.8  ", 7.8),
    ("", None), ("-", None), ("--", None), ("n/a", None), ("N/A", None),
    ("null", None), ("None", None), ("?", None),
    ("llama", None), ("10/1/2025", None),
    (5, 5.0), (5.5, 5.5), (None, None), ([], None),
])
def test_which_cells_are_numbers(cell, expected):
    assert _as_number(cell) == expected


def test_a_flag_is_never_a_score():
    """A bool is an int in Python. Scoring one turns `Is Thinking Model` into a metric worth 1.0."""
    assert _as_number(True) is None
    assert _as_number(False) is None


@pytest.mark.parametrize(("cell", "expected"), [
    ("36.5", 1), ("27.93", 2), ("36.500", 3), ("36", 0), ("  36.5 ", 1),
    (7, 0), (7.5, None), (True, None), ("llama", None), (None, None),
])
def test_the_precision_a_figure_was_quoted_at(cell, expected):
    """Read off the string, because `36.50` and `36.5` are the same float and different claims."""
    assert quoted_precision(cell) == expected


# ── the adapter ───────────────────────────────────────────────────────────────────────


def test_the_adapter_owns_a_leaderboard_document(tmp_path):
    doc = load_document(write_csv(tmp_path, HARBINGER))
    assert detect(doc) is LeaderboardRowAdapter


def test_the_adapter_declines_anything_else():
    assert LeaderboardRowAdapter.detects({"results": {}, "configs": {}, "versions": {}}) is False
    assert LeaderboardRowAdapter.detects({}) is False


def test_a_real_row_normalises(tmp_path):
    n = normalise(load_document(write_csv(tmp_path, HARBINGER)))
    assert n["harness"] == "public leaderboard row"
    assert n["artefact_kind"] == "leaderboard-row"
    assert n["model"] == "LatitudeGames/Harbinger-24B"
    assert n["leaderboard"] == "board"
    assert n["metrics"]["UGI"]["value"] == 36.5
    assert n["metrics"]["NatInt"]["value"] == 27.93
    assert n["metrics"]["UGI"]["quoted_decimals"] == 1
    assert n["metrics"]["NatInt"]["quoted_decimals"] == 2


def test_no_sample_size_and_no_uncertainty_are_invented(tmp_path):
    """The property this whole adapter exists to hold. See the module docstring."""
    n = normalise(load_document(write_csv(tmp_path, HARBINGER)))
    assert n["metrics"], "there should be metrics to check this on"
    for key, metric in n["metrics"].items():
        assert metric["n"] is None, f"{key} was given a sample size the source does not publish"
        assert metric["stderr"] is None, f"{key} was given an uncertainty nobody published"
    assert n["any_metric_carries_n"] is False
    assert n["any_metric_carries_uncertainty"] is False


def test_no_units_are_claimed(tmp_path):
    """A 0 to 100 column and a 0 to 10 column sit side by side and nothing says which is which.

    Asserting `proportion` would make the impossible-proportion check fire on every row of a
    0 to 100 column, so the honest answer is to claim nothing and let that check skip.
    """
    n = normalise(load_document(write_csv(tmp_path, HARBINGER)))
    assert all(m["units"] is None for m in n["metrics"].values())


def test_identity_and_attribute_columns_are_not_scores(tmp_path):
    n = normalise(load_document(write_csv(tmp_path, HARBINGER)))
    assert "author/model_name" not in n["metrics"]
    assert "Model Link" not in n["metrics"]
    assert "Test Date" not in n["metrics"]
    assert "Is Thinking Model" not in n["metrics"]
    assert "author/model_name" in n["attributes"]
    assert "Test Date" in n["attributes"]


def test_a_dash_cell_is_an_attribute_and_not_a_zero(tmp_path):
    """`Writing` is `-` on this row. Reading it as 0.0 would invent a score of nought."""
    n = normalise(load_document(write_csv(tmp_path, ABLITERATED)))
    assert "Writing" not in n["metrics"]
    assert n["attributes"]["Writing"] == "-"


def test_the_gap_a_ranking_cannot_support_is_visible(tmp_path):
    """Both real rows, side by side, with the capability column moving the other way."""
    a = normalise(load_document(write_csv(tmp_path, HARBINGER, name="a.csv")))
    b = normalise(load_document(write_csv(tmp_path, ABLITERATED, name="b.csv")))
    assert b["metrics"]["UGI"]["value"] > a["metrics"]["UGI"]["value"]
    assert b["metrics"]["NatInt"]["value"] < a["metrics"]["NatInt"]["value"]
    # And neither carries anything that would let the difference be called real.
    assert not a["any_metric_carries_n"] and not b["any_metric_carries_n"]


def test_a_row_with_no_identifiable_name_still_normalises(tmp_path):
    p = write_csv(tmp_path, "4.0,5.0", header="UGI,W/10")
    n = normalise(load_document(p))
    assert n["model"] is None
    assert n["metric_count"] == 2


def test_a_row_of_only_labels_reports_no_metrics(tmp_path):
    p = write_csv(tmp_path, "llama,-", header="author/model_name,Writing")
    n = normalise(load_document(p))
    assert n["metric_count"] == 0
    assert n["eval_split"] is None and n["limit"] is None


# ── choosing a row, which is a refusal and not a guess ────────────────────────────────


def test_a_single_row_file_needs_no_selection(tmp_path):
    doc = load_document(write_csv(tmp_path, HARBINGER))
    assert doc["rows_available"] == 1


def test_many_rows_with_no_selection_is_refused(tmp_path):
    p = write_csv(tmp_path, HARBINGER, ABLITERATED)
    with pytest.raises(LoaderError, match="nothing says which one to check"):
        load_document(p)


def test_the_refusal_names_the_columns_available(tmp_path):
    p = write_csv(tmp_path, HARBINGER, ABLITERATED)
    with pytest.raises(LoaderError, match="author/model_name"):
        load_document(p)


def test_an_exact_selection_wins(tmp_path):
    p = write_csv(tmp_path, HARBINGER, ABLITERATED)
    doc = load_document(p, row_select="LatitudeGames/Harbinger-24B")
    assert doc["row"]["UGI"] == "36.5"


def test_selection_is_case_insensitive(tmp_path):
    p = write_csv(tmp_path, HARBINGER, ABLITERATED)
    doc = load_document(p, row_select="latitudegames/harbinger-24b")
    assert doc["row"]["UGI"] == "36.5"


def test_a_partial_selection_matching_one_row_wins(tmp_path):
    p = write_csv(tmp_path, HARBINGER, ABLITERATED)
    doc = load_document(p, row_select="biprojected")
    assert doc["row"]["UGI"] == "43.6"


def test_a_selection_matching_nothing_is_refused(tmp_path):
    p = write_csv(tmp_path, HARBINGER)
    with pytest.raises(LoaderError, match="matches no row"):
        load_document(p, row_select="Qwen/Qwen3-8B")


def test_a_partial_selection_matching_several_rows_is_refused(tmp_path):
    """`Harbinger` is in both names, so it does not identify one of them."""
    p = write_csv(tmp_path, HARBINGER, ABLITERATED)
    with pytest.raises(LoaderError, match="matches 2 rows"):
        load_document(p, row_select="Harbinger")


def test_an_exact_selection_matching_several_rows_is_refused(tmp_path):
    """A ranking listing one model twice is listing two settings, and those are two measurements."""
    p = write_csv(tmp_path, HARBINGER, HARBINGER)
    with pytest.raises(LoaderError, match="matches 2 rows exactly"):
        load_document(p, row_select="LatitudeGames/Harbinger-24B")


def test_the_ambiguous_refusal_names_candidates(tmp_path):
    p = write_csv(tmp_path, HARBINGER, ABLITERATED)
    with pytest.raises(LoaderError, match="LatitudeGames/Harbinger-24B"):
        load_document(p, row_select="Harbinger")


def test_a_candidate_name_falls_back_when_no_column_looks_like_a_path(tmp_path):
    p = write_csv(tmp_path, "alpha,1.0", "alpha,2.0", header="label,UGI")
    with pytest.raises(LoaderError, match="matches 2 rows exactly"):
        load_document(p, row_select="alpha")


def test_the_candidate_list_names_rows_that_carry_no_path_like_column(tmp_path):
    """The ambiguous-partial refusal has to name candidates even with no `org/model` column."""
    p = write_csv(tmp_path, "alpha-one,1.0", "alpha-two,2.0", header="label,UGI")
    with pytest.raises(LoaderError, match="alpha-one"):
        load_document(p, row_select="alpha")


def test_a_row_whose_every_cell_is_blank_still_produces_a_name(tmp_path):
    """`_identify` must return something printable rather than raising inside a refusal."""
    p = write_csv(tmp_path, ",", ",", header="label,other")
    with pytest.raises(LoaderError, match="matches no row"):
        load_document(p, row_select="zzz")


# ── refusing a file that is not one ───────────────────────────────────────────────────


def test_a_header_with_no_rows_is_refused(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text(HEADER + "\n", encoding="utf-8")
    with pytest.raises(LoaderError, match="no rows under it"):
        load_document(p)


def test_a_file_with_no_header_is_refused(tmp_path):
    p = tmp_path / "blank.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(LoaderError, match="no header line"):
        load_document(p)


def test_a_csv_that_is_not_text_is_refused(tmp_path):
    p = tmp_path / "binary.csv"
    p.write_bytes(b"\xff\xfe\x00\x01 not utf-8 \xff")
    with pytest.raises(LoaderError, match="not text this tool can decode"):
        load_document(p)


def test_an_unreadable_csv_is_refused(tmp_path):
    with pytest.raises(LoaderError, match="could not read it"):
        load_document(tmp_path / "absent.csv")


def test_a_byte_order_mark_does_not_corrupt_the_first_column(tmp_path):
    """The real defect this decoding choice exists for. See the loader's comment.

    Under plain utf-8 the mark becomes part of the first column's name, a lookup on the plain name
    matches nothing, and the zero reads as a fact rather than as a bug in the query.
    """
    p = tmp_path / "bom.csv"
    p.write_bytes(b"\xef\xbb\xbf" + (HEADER + "\n" + HARBINGER + "\n").encode("utf-8"))
    doc = load_document(p)
    assert "author/model_name" in doc["row"], "the mark was not stripped from the first column"
    n = normalise(doc)
    assert n["model"] == "LatitudeGames/Harbinger-24B"


def test_a_tab_separated_export_is_read(tmp_path):
    p = tmp_path / "board.tsv"
    p.write_text("author/model_name\tUGI\nLatitudeGames/Harbinger-24B\t36.5\n", encoding="utf-8")
    n = normalise(load_document(p))
    assert n["metrics"]["UGI"]["value"] == 36.5


def test_too_many_rows_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("senbonzakura_check.loaders.MAX_ROWS", 2)
    p = write_csv(tmp_path, HARBINGER, ABLITERATED, HARBINGER)
    with pytest.raises(LoaderError, match="not a leaderboard export"):
        load_document(p)


def test_the_row_cap_is_generous_enough_for_a_real_board():
    """1,326 rows is the real file. A cap below that would refuse the thing it exists to read."""
    assert MAX_ROWS > 100_000


def test_a_json_artefact_still_reaches_the_json_reader(tmp_path):
    """THE REGRESSION THAT MATTERS. The loader now sits in front of every read, so the three
    existing adapters have to keep working through it. A new branch that stole JSON documents
    would break the formats this package was built for, and the tests for those live elsewhere
    and would not name this loader as the cause.
    """
    import json

    doc = {"results": {"gsm8k": {"acc,none": 0.5, "alias": "gsm8k"}},
           "configs": {"gsm8k": {"test_split": "test"}}, "versions": {"gsm8k": 1.0},
           "config": {"model": "hf", "limit": None}}
    p = tmp_path / "lm-eval.json"
    p.write_text(json.dumps(doc), encoding="utf-8")

    loaded = load_document(p)
    assert loaded == doc, "a JSON artefact must arrive unchanged, not wrapped"
    n = normalise(loaded)
    assert n["harness"] == "lm-evaluation-harness"


def test_a_json_file_without_a_json_suffix_still_reaches_the_json_reader(tmp_path):
    """Suffix is not how JSON is detected, and must not become how it is."""
    import json

    p = tmp_path / "result.log"
    p.write_text(json.dumps({"results": {"t": {"acc,none": 1.0}}, "configs": {}, "versions": {}}),
                 encoding="utf-8")
    assert normalise(load_document(p))["harness"] == "lm-evaluation-harness"


# ── end to end, through the checks ────────────────────────────────────────────────────


def test_the_no_uncertainty_check_fires_end_to_end(tmp_path):
    from senbonzakura_check.cli import inspect_file
    from senbonzakura_check.registry import load_checks

    p = write_csv(tmp_path, HARBINGER)
    findings, _skipped, problem = inspect_file(p, load_checks())
    assert problem is None
    assert "a-public-score-with-nothing-behind-it" in [f.check_id for f in findings]


def test_a_row_with_no_scores_does_not_fire_the_uncertainty_check(tmp_path):
    from senbonzakura_check.cli import inspect_file
    from senbonzakura_check.registry import load_checks

    p = write_csv(tmp_path, "llama,-", header="author/model_name,Writing")
    findings, _skipped, problem = inspect_file(p, load_checks())
    assert problem is None
    assert "a-public-score-with-nothing-behind-it" not in [f.check_id for f in findings]


def test_the_pair_check_fires_on_two_rows(tmp_path):
    from senbonzakura_check.cli import inspect_pair
    from senbonzakura_check.registry import load_checks

    a = write_csv(tmp_path, HARBINGER, name="a.csv")
    b = write_csv(tmp_path, ABLITERATED, name="b.csv")
    findings, _skipped, problem = inspect_pair(a, b, load_checks())
    assert problem is None
    assert "two-rankings-subtracted-with-no-sample-size-on-either" in [
        f.check_id for f in findings]


def test_a_pair_with_one_unreadable_arm_is_a_problem_not_a_clean_result(tmp_path):
    from senbonzakura_check.cli import inspect_pair
    from senbonzakura_check.registry import load_checks

    a = write_csv(tmp_path, HARBINGER, name="a.csv")
    findings, _skipped, problem = inspect_pair(a, tmp_path / "absent.csv", load_checks())
    assert findings == []
    assert problem is not None


def test_the_cli_refuses_a_whole_board_without_a_row(tmp_path, capsys):
    from senbonzakura_check.cli import main

    p = write_csv(tmp_path, HARBINGER, ABLITERATED)
    code = main([str(p)])
    out = capsys.readouterr().out
    assert "nothing says which one to check" in out
    assert "NOTHING WAS CHECKED" in out
    assert code == 2, "a named file nobody could read must not exit 0"


def test_the_cli_accepts_a_row_flag(tmp_path, capsys):
    from senbonzakura_check.cli import main

    p = write_csv(tmp_path, HARBINGER, ABLITERATED)
    code = main([str(p), "--row", "LatitudeGames/Harbinger-24B", "--quiet"])
    out = capsys.readouterr().out
    assert "a-public-score-with-nothing-behind-it" in out
    assert code == 1, "a finding exits 1"
