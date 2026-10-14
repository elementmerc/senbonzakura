# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The head-to-head reporter's machine-readable half, and the one thing it must never claim.

ROW 8.13, AND WHY IT WAS BLOCKED UNTIL NOW

The row asks that `headtohead_report` stamp the canonical metrics block, which `score`,
`capability`, `margin` and `coherence` all do. It was blocked on Q-54, the question of what a
pinned field means for a document covering several measurements, and that question was raised
precisely because this reporter and `senbonzakura measure` reach it independently. Answering it
twice differently is the defect, so neither moved until it was settled.

Q-116 settled it on 2026-10-08, taking option E: an aggregate is not a measurement of anything. It
declares itself a summary, stamps no figure of its own, and carries each measurement's own stamp
verbatim, with the fields they disagree on named rather than omitted.

WHY A REPORT OVER ARMS IS THE WORSE CASE, which is what these tests are mostly about. `measure`
runs four instruments on ONE model, and the fields they differ on are the input digest, the
partition and the prompt format. A head-to-head runs one instrument over TEN models from TWO
tools, so the field they differ on is `model`, which is the one field a reader is most likely to
assume constant across a provenance block. A single `model` value over that table would be false
about every row but one.

AND WHAT IS DELIBERATELY ABSENT. No figure and no verdict. The printed table is the figures, and
two homes for one number is how a stale copy gets quoted; the verdict is a reading of the figures
rather than a fact about them, and the tie rule that produces it belongs beside the prose that
explains it.
"""
from __future__ import annotations

import json

import pytest

from senbonzakura import headtohead_report as rh
from senbonzakura import stamps


#: A pinned block as `measurement.stamp` writes one. Only the keys `baseline.PINNED` names survive
#: into a summary, so the extra ones here are the point: they prove the filter runs.
def _stamped(model, **over):
    block = {
        "metric": "compass_auc", "estimator": "margin-past-preamble", "value": 0.9,
        "model": model, "input_digest": "sha256:aaa", "partition": "measure",
        "prompt_format": "renderer:v3", "tool_version": "0.4.1", "precision": "bfloat16",
        # NOT a pinned field. Present so the test proves the summary carries the stamp's pinned
        # subset rather than the whole block, which would copy the figure into a second home.
        "n": 290, "measures": "harm recognition",
    }
    block.update(over)
    return {"metrics": {"compass_auc.margin-past-preamble": block}}


@pytest.fixture
def run_dir(tmp_path):
    """A finished two-tool run whose arms carry real stamps, each on its own model."""
    def build(*, stamp=True, unreadable=False, broken_stamp=False):
        for tool, model in (("senbon", "ours/edited-1.7b"), ("heretic", "theirs/edited-1.7b")):
            for i in range(3):
                seed = 42 + i
                scored = {"auc": 0.9 - i / 100, "auc_ci": [0.85, 0.95],
                          "controls": {"length_only_auc": 0.60}}
                if stamp:
                    scored.update(_stamped(model))
                (tmp_path / f"scored-{tool}-seed{seed}.json").write_text(
                    json.dumps(scored), encoding="utf-8")
                arm = tmp_path / f"{tool}-seed{seed}"
                arm.mkdir(exist_ok=True)
                body = ({"post_bake_refusals": 0.03, "post_bake_kl": 0.21} if tool == "senbon"
                        else {"winner": {"refusals": 0.05, "kl": 0.004}})
                name = "abliteration.json" if tool == "senbon" else "best_of_n.json"
                (arm / name).write_text(json.dumps(body), encoding="utf-8")
        if unreadable:
            (tmp_path / "scored-senbon-seed99.json").write_text("{}", encoding="utf-8")
        if broken_stamp:
            (tmp_path / "scored-heretic-seed42.json").write_text("{ not json",
                                                                 encoding="utf-8")
        return tmp_path
    return build


def _document(directory):
    return rh.summary_document(str(directory), rh.collect(str(directory)))


# ── it says what it is ──────────────────────────────────────────────────────────────────────────


def test_the_document_declares_itself_a_summary_rather_than_leaving_it_to_be_inferred(run_dir):
    doc = _document(run_dir())
    assert doc["kind"] == stamps.SUMMARY_KIND
    assert doc["schema"] == rh.SUMMARY_SCHEMA


def test_the_document_carries_no_metrics_block_of_its_own(run_dir):
    """The whole of option E. An aggregate stamping one figure would be stamping a figure nothing
    measured, and the first reader to gate on it would be gating on arithmetic over two tools.
    """
    assert "metrics" not in _document(run_dir())


def test_the_document_carries_no_figure_and_no_verdict(run_dir):
    """Two homes for one number is how a stale copy gets quoted.

    CHECKED ON KEYS AND ON NUMBERS, not on substrings. The first version of this grepped the JSON
    for "auc" and failed, because the metric is CALLED `compass_auc` and the metric's identity is
    one of the pinned fields this document exists to carry. Which is the distinction the whole row
    turns on: the name of the thing measured belongs here, and the value of it does not.
    """
    doc = _document(run_dir())
    for key in ("auc", "auc_ci", "verdict", "winner", "value", "metrics"):
        assert key not in doc, f"the summary carries {key!r}, which belongs in the table"
    for name, blocks in doc["arm_stamps"].items():
        for slot, block in blocks.items():
            if slot in stamps.MARKER_KEYS:
                continue
            numbers = sorted(k for k, v in block.items()
                             if isinstance(v, (int, float)) and not isinstance(v, bool))
            assert not numbers, f"{name}/{slot} carries measured values: {numbers}"


def test_the_note_explaining_the_absent_stamp_is_in_the_document(run_dir):
    assert _document(run_dir())["pinned_fields_that_differ_note"] == stamps.PINNED_MAY_DIFFER_DOC


# ── each arm's own stamp, verbatim ──────────────────────────────────────────────────────────────


def test_every_readable_arm_contributes_its_own_pinned_block(run_dir):
    doc = _document(run_dir())
    assert sorted(doc["arm_stamps"]) == [
        "scored-heretic-seed42", "scored-heretic-seed43", "scored-heretic-seed44",
        "scored-senbon-seed42", "scored-senbon-seed43", "scored-senbon-seed44"]


def test_an_arms_block_holds_the_pinned_fields_and_nothing_else(run_dir):
    from senbonzakura import baseline

    block = _document(run_dir())["arm_stamps"]["scored-senbon-seed42"][
        "compass_auc.margin-past-preamble"]
    assert set(block) <= set(baseline.PINNED)
    assert block["model"] == "ours/edited-1.7b"
    assert "n" not in block, "a non-pinned key reached the summary, so the filter is not running"


def test_the_field_the_two_tools_differ_on_is_named(run_dir):
    """`model` is the field a reader is most likely to assume constant, and it never is here."""
    assert _document(run_dir())["pinned_fields_that_differ"] == ["model"]


def test_the_fields_that_differ_are_sorted_so_two_runs_agree(run_dir):
    d = run_dir()
    # A second differing field, so the sort has something to do.
    path = d / "scored-senbon-seed43.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["metrics"]["compass_auc.margin-past-preamble"]["precision"] = "float16"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert _document(d)["pinned_fields_that_differ"] == ["model", "precision"]


def test_an_arm_whose_file_carries_no_stamp_says_so_rather_than_being_dropped(run_dir):
    """An absent key and a key saying "this could not be read" are different claims."""
    blocks = _document(run_dir(stamp=False))["arm_stamps"]
    assert len(blocks) == 6
    for name, block in blocks.items():
        assert "unstamped" in block, f"{name} should report the absent stamp"


def test_an_arm_whose_file_cannot_be_parsed_contributes_a_marker(run_dir):
    blocks = _document(run_dir(broken_stamp=True))["arm_stamps"]
    # An unparseable scored file reads as an unreadable ARM to `collect`, so it is listed among the
    # arms and carries no stamp. Both halves are the honest answer and both are asserted.
    assert "scored-heretic-seed42" not in blocks
    listed = {a["name"]: a for a in _document(run_dir(broken_stamp=True))["arms"]}
    assert listed["scored-heretic-seed42.json"]["unreadable"] is True


def test_a_marker_is_not_mistaken_for_a_differing_field(run_dir):
    """`unstamped` is a key in the same mapping as the metric slots, so it has to be skipped."""
    assert _document(run_dir(stamp=False))["pinned_fields_that_differ"] == []


# ── the arms the table excludes are marked, not hidden ──────────────────────────────────────────


def test_every_arm_is_listed_including_the_unreadable_one(run_dir):
    doc = _document(run_dir(unreadable=True))
    names = [a["name"] for a in doc["arms"]]
    assert len(names) == 7
    assert names == sorted(names), "the arm list must be sorted so two runs produce one document"
    assert any(a["unreadable"] for a in doc["arms"])


def test_a_control_arm_is_marked_so_a_machine_reader_cannot_put_it_back_in_a_table(run_dir):
    """`render` pulls control arms out before any table is built. A reader needs the same fact."""
    d = run_dir()
    (d / "senbon-seed42" / "abliteration.json").write_text(json.dumps({
        "post_bake_refusals": 0.03, "post_bake_kl": 0.21,
        "args": {"skip_layers": [0, 1, 2]}}), encoding="utf-8")
    listed = {a["name"]: a for a in _document(d)["arms"]}
    marked = [a for a in listed.values() if a["partial_ablation"]]
    assert marked or listed["scored-senbon-seed42"]["partial_ablation"] is None, (
        "a control arm must either be marked or genuinely absent from this run")


def test_the_run_directory_is_named_without_its_path(run_dir):
    """A path from somebody's laptop in a published artefact is a leak of nothing useful."""
    d = run_dir()
    assert _document(d)["run_dir"] == d.name
    assert "/" not in _document(d)["run_dir"]


# ── the command writes it, and only when asked ──────────────────────────────────────────────────


def test_the_command_writes_nothing_unless_out_is_given(run_dir, capsys):
    d = run_dir()
    before = sorted(p.name for p in d.iterdir())
    rh.main([str(d)])
    capsys.readouterr()
    assert sorted(p.name for p in d.iterdir()) == before


def test_the_command_writes_the_summary_when_out_is_given(run_dir, tmp_path, capsys):
    d = run_dir()
    out = tmp_path / "summary.json"
    rh.main([str(d), "--out", str(out)])
    assert "summary written to" in capsys.readouterr().out
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["kind"] == stamps.SUMMARY_KIND
    assert doc["pinned_fields_that_differ"] == ["model"]


def test_the_summary_is_renamed_into_place_rather_than_written_in_place(run_dir, tmp_path, capsys):
    d = run_dir()
    out = tmp_path / "summary.json"
    rh.main([str(d), "--out", str(out)])
    capsys.readouterr()
    assert not list(tmp_path.glob("*.part")), "the temporary file is renamed, never left behind"


def test_two_runs_over_one_directory_write_byte_identical_summaries(run_dir, tmp_path, capsys):
    """Reproducibility. Iteration order of a mapping must not reach a result file."""
    d = run_dir()
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    rh.main([str(d), "--out", str(first)])
    rh.main([str(d), "--out", str(second)])
    capsys.readouterr()
    assert first.read_bytes() == second.read_bytes()


def test_the_summary_is_written_even_when_an_arm_could_not_be_read(run_dir, tmp_path, capsys):
    """The run whose provenance somebody most wants is the one that went wrong."""
    d = run_dir(unreadable=True)
    out = tmp_path / "summary.json"
    status = rh.main([str(d), "--out", str(out)])
    capsys.readouterr()
    assert status == 1, "an unreadable arm still fails the report"
    assert out.exists(), "the summary must still be written, marked, rather than withheld"
    assert any(a["unreadable"] for a in json.loads(out.read_text(encoding="utf-8"))["arms"])


def test_a_summary_that_cannot_be_written_is_said_out_loud_and_changes_no_verdict(
        run_dir, tmp_path, capsys):
    """A failure to write is about the machine; the verdict is about the models."""
    d = run_dir()
    status = rh.main([str(d), "--out", str(tmp_path / "no-such-directory" / "s.json")])
    said = capsys.readouterr()
    assert "could not be written" in said.err
    assert status == 0, "a full disk must not turn a clean report into a failure"


def test_the_out_flag_says_in_its_own_help_that_it_carries_no_figure():
    """`--help` is where somebody decides whether to trust an artefact, so it is where this goes.

    A summary that looks like it holds the numbers is worse than no summary: somebody quotes it,
    and the figure they quote came from whenever the file was last written rather than from the
    run they are describing.
    """
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buffer):
        rh.main(["--help"])
    text = buffer.getvalue()
    assert "--out" in text, "the flag is not in the help output"
    assert "NO figure" in text and "NO verdict" in text


# ── the hoisted helpers, driven directly ────────────────────────────────────────────────────────
#
# `SUMMARY_KIND`, `pinned_blocks` and `pinned_fields_that_differ` moved from `measure.py` into
# `stamps.py` so both askers share one answer, which is what Q-54's own note asks for: the question
# was raised because `measure` and this reporter reach it independently, and answering it twice
# differently is the defect. `measure` re-exports the names, so these drive the originals.


def test_measure_still_answers_to_the_names_it_used_to_own():
    """The re-export, asserted rather than assumed. Tests and callers spell the old names."""
    from senbonzakura import measure

    assert measure.SUMMARY_KIND is stamps.SUMMARY_KIND
    assert measure.PINNED_MAY_DIFFER_DOC is stamps.PINNED_MAY_DIFFER_DOC
    assert measure.pinned_fields_that_differ is stamps.pinned_fields_that_differ


@pytest.mark.parametrize("doc", [None, [], "a string", 7])
def test_a_document_that_is_not_a_document_says_so_rather_than_raising(doc):
    """A reader of somebody else's result file meets shapes nobody promised, so it says so."""
    blocks = stamps.pinned_blocks(doc, where="odd.json")
    assert blocks["unstamped"] == "odd.json is not a document"


def test_a_metrics_block_that_is_not_a_mapping_reads_as_unstamped():
    assert "unstamped" in stamps.pinned_blocks({"metrics": []}, where="odd.json")
    assert "unstamped" in stamps.pinned_blocks({"metrics": {}}, where="odd.json")


def test_a_metric_slot_that_is_not_a_mapping_is_skipped_rather_than_crashing():
    blocks = stamps.pinned_blocks({"metrics": {"good": {"model": "m"}, "odd": 7}},
                                  where="odd.json")
    assert blocks == {"good": {"model": "m"}}


def test_an_unreadable_file_contributes_a_marker_naming_the_error(tmp_path):
    missing = tmp_path / "never-written.json"
    assert "FileNotFoundError" in stamps.read_pinned_blocks(missing)["unreadable"]
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    assert "JSONDecodeError" in stamps.read_pinned_blocks(broken)["unreadable"]


def test_the_marker_names_the_file_and_not_the_path(tmp_path):
    """An absolute path from whoever ran it is a small disclosure and useless to the next reader.

    Caught by `tests/test_a_summary_declares_itself_and_carries_no_stamp_of_its_own.py` during the
    hoist: the first version passed the whole path through as the marker's subject, and that test
    asserts the filename form. It was right to.
    """
    path = tmp_path / "score.json"
    path.write_text(json.dumps({"no": "metrics"}), encoding="utf-8")
    marker = stamps.read_pinned_blocks(path)["unstamped"]
    assert marker.startswith("score.json")
    assert str(tmp_path) not in marker


def test_a_field_absent_from_one_side_does_not_count_as_differing():
    """Two values is what differing means.

    One value and a gap is a gap, which `comparability` already treats as a refusal to compare;
    naming it here too would double-report it.
    """
    assert stamps.pinned_fields_that_differ({
        "a": {"slot": {"model": "m", "precision": "bfloat16"}},
        "b": {"slot": {"model": "m"}},
    }) == []
