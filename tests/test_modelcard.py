# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The card beside the weights states what the artefacts support and nothing else.

WHY THIS FILE EXISTS

Abliterated models get published with a sentence like "minimal degradation" and nothing behind it.
The field ships on a claimed one to three percent with no benchmarks; a comparative study puts one
family at eight points on grade-school arithmetic. Neither side offers an interval and nobody can
check either.

Everything needed to do better is already on disk when a run finishes. What was missing is the
page, and the discipline that a page is where overclaiming happens.

THE FOUR RULES UNDER TEST

A number not in an artefact does not appear. A rate the sample cannot carry is printed as counts.
An interval spanning zero is written as "not distinguishable from no change" rather than as a
delta with a caveat further down. And a section with no evidence says NOT MEASURED in those words,
because an absent section reads as nothing to report and that is a different claim.
"""
import json

import pytest

from senbonzakura import modelcard


def _abl(**over):
    d = {"model": "M", "method": "searched", "baseline_refusal": 0.4, "post_bake_refusal": 0.0,
         "post_bake_kl": 0.12, "track": "default"}
    d.update(over)
    return d


def _cap(n=200, graded=190, correct=120, change=None, **over):
    d = {"task": "numeric", "eval": "gsm8k",
         "summary": {"n": n, "graded": graded, "correct": correct, "wrong": graded - correct,
                     "indeterminate": n - graded},
         "change": change}
    d.update(over)
    return d


def _text(**kw):
    return "\n".join(modelcard.build(**kw))


# ── the rule that matters most: absence is stated ────────────────────────────────────

def test_a_card_with_no_capability_evidence_declares_it_unmeasured():
    """THE RULE THIS FILE EXISTS FOR. Omitting the section would read as nothing to report."""
    text = _text(abl=_abl())
    assert "NOT MEASURED" in text
    assert "Capability" in text


def test_the_missing_capability_section_explains_why_the_other_numbers_do_not_cover_it():
    """A reader who sees a low KL and no capability section will assume the KL covered it."""
    text = _text(abl=_abl())
    assert "cannot see reasoning loss" in text
    assert "no claim about capability is supported" in text


def test_every_section_appears_even_with_no_artefacts_at_all():
    text = _text()
    for heading in ("What was done", "Refusal", "Capability", "Corpus", "Reproducing"):
        assert heading in text, f"{heading} was dropped rather than declared"


def test_a_run_with_no_command_says_it_cannot_be_reproduced():
    assert "NOT RECORDED" in _text(abl=_abl())
    assert "senbonzakura --method x" in _text(abl=_abl(), command="senbonzakura --method x")


# ── rates the sample cannot carry ────────────────────────────────────────────────────

def test_a_small_capability_run_is_reported_as_counts_not_a_rate():
    """The reporting floor reaches the card, or the card is where the defect comes back."""
    text = _text(cap=_cap(n=5, graded=5, correct=3))
    assert "3/5" in text
    assert "not stated as a rate" in text
    assert "60.0%" not in text


def test_a_large_run_gets_its_rate_with_the_counts_beside_it():
    text = _text(cap=_cap())
    assert "120/190" in text
    assert "95% CI" in text


# ── the change, and what an overlapping interval must read as ────────────────────────

def test_a_real_drop_is_stated_with_its_interval():
    change = {"compared_on": 185, "delta_accuracy": -0.078, "delta_ci": [-0.121, -0.036],
              "distinguishable_from_zero": True, "items_broken": 20, "items_fixed": 6}
    text = _text(cap=_cap(change=change))
    assert "-7.8%" in text
    assert "95% CI" in text


def test_an_interval_spanning_zero_is_not_written_as_a_delta():
    """A point estimate with a caveat three lines below is how a null result gets quoted as a
    finding. The words come first here, and the number after them.
    """
    change = {"compared_on": 100, "delta_accuracy": -0.01, "delta_ci": [-0.06, 0.04],
              "distinguishable_from_zero": False, "items_broken": 5, "items_fixed": 4}
    text = _text(cap=_cap(change=change))
    assert "not distinguishable from no change" in text
    line = next(ln for ln in text.splitlines() if "change in accuracy" in ln)
    assert "not distinguishable" in line, "the caveat must be on the same line as the number"


def test_items_moving_both_ways_is_called_out_as_a_different_finding():
    change = {"compared_on": 100, "delta_accuracy": 0.0, "delta_ci": [-0.05, 0.05],
              "distinguishable_from_zero": False, "items_broken": 10, "items_fixed": 10}
    assert "answers differently" in _text(cap=_cap(change=change))


def test_a_capability_run_with_no_reference_says_it_is_not_a_cost():
    """An absolute score is not a statement about what the edit did, and a reader will take it
    as one unless told.
    """
    text = _text(cap=_cap())
    assert "not a statement about what the edit cost" in text


def test_a_high_indeterminate_count_is_surfaced_beside_the_accuracy():
    text = _text(cap=_cap(n=200, graded=100, correct=60))
    assert "100 of 200 answers could not be graded" in text


# ── loading, and refusing to make something up ───────────────────────────────────────

def test_an_artefact_that_was_not_supplied_is_a_gap(tmp_path):
    """The one case that is genuinely a gap: nobody named a file, so nothing is missing."""
    assert modelcard.load("") is None
    assert modelcard.load(None) is None


def test_an_artefact_that_was_named_and_is_absent_refuses(tmp_path):
    """REPLACES `test_a_missing_artefact_is_a_gap_rather_than_an_error`, 2026-09-27.

    That test asserted `load(tmp_path / "nope.json") is None`, and nothing recorded why beyond a
    one-line docstring saying a missing file is a gap. A surface audit against a built wheel showed
    what it cost: `--abliteration notjson.txt` produced a complete, publishable model card that said
    the evidence was NOT MEASURED, and exited 0. A reader concludes the run wrote an empty artefact.

    The distinction the old test flattened is between "no path was given", which is a gap, and "a
    path was given and cannot be used", which is the user naming a file they believe in. The first
    still returns None, above. The second stops.
    """
    with pytest.raises(SystemExit) as e:
        modelcard.load(tmp_path / "nope.json")
    # NORMALISED, because these messages are wrapped now and a substring can straddle a line break.
    # This assertion first read `match="is not a file"` and failed on "is\nnot a file", which is the
    # same trap any consumer grepping this output can fall into.
    assert "is not a file" in " ".join(str(e.value).split())


def test_an_unreadable_artefact_refuses_rather_than_reporting_nothing_measured(tmp_path):
    """REPLACES `test_an_unreadable_artefact_is_a_gap_rather_than_a_crash`, same reason.

    "Rather than a crash" was the right instinct and None was the wrong destination: reporting NOT
    MEASURED says the run measured nothing, when what happened is that this file could not be read.
    Those are different sentences and only one of them is true.
    """
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        modelcard.load(p)
    flat = " ".join(str(e.value).split())
    assert "not readable JSON" in flat
    assert "NOT MEASURED" in flat, (
        "the refusal should say what it is NOT doing, because the old behaviour is what the reader "
        "is expecting to see")


def test_the_command_refuses_to_build_a_card_from_nothing():
    """A card with no artefacts behind it is a template, and this exists to stop those being
    published.
    """
    with pytest.raises(SystemExit, match="nothing to report on"):
        modelcard.main([])


def test_the_command_writes_a_file_when_asked(tmp_path):
    abl = tmp_path / "a.json"
    abl.write_text(json.dumps(_abl()), encoding="utf-8")
    out = tmp_path / "card.md"
    # --base-licence is required now (panel finding C6): the card states that the base model's
    # licence governs the weights, and it must name it rather than assert an unnamed one.
    assert modelcard.main(["--abliteration", str(abl), "--out", str(out),
                           "--base-licence", "apache-2.0"]) == 0
    assert "Abliteration report" in out.read_text(encoding="utf-8")


def test_the_card_never_invents_a_number_that_is_not_in_an_artefact():
    """The load-bearing property. Every figure printed has to trace to a key that was supplied."""
    text = _text(abl={"model": "M", "method": "single-pass"})
    for absent in ("refusal before", "KL drift", "broken output"):
        assert absent not in text, f"{absent} was printed from an artefact that did not carry it"


def test_the_card_says_the_base_licence_is_unchanged_and_the_refusals_are_gone():
    """THE ONE ARTEFACT THAT TRAVELS, and it carried none of this.

    The repository states both things carefully: a base model's licence is not ours to loosen,
    and abliteration removes safety guardrails wholesale. All of it reaches a reader of the
    repository. None of it reached the HuggingFace page of a checkpoint somebody abliterated,
    which is the only artefact a downstream user of those weights will ever see.
    """
    from senbonzakura import modelcard

    page = "\n".join(modelcard.build())
    assert "base model's licence" in page
    assert "does not create a new work with a new licence" in page
    assert "removed on purpose" in page
    assert "senbonzakura" in page, "the card must say what generated it"


def test_the_licence_section_needs_no_inputs():
    """A field somebody can leave blank is a field that gets left blank, so there is no field.

    `build()` with no abliteration and no capability artefact still carries the whole statement.
    """
    from senbonzakura import modelcard

    assert "licence and use" in modelcard.SECTIONS
    page = "\n".join(modelcard.build(abl=None, cap=None, command=None))
    assert "not ours to loosen" in page or "base model's licence" in page


# ── the licence, which the card asserted and never named (panel finding C6) ───────────
#
# The card told a reader of the published weights that the base model's licence governs them, and
# then never said WHICH licence, never linked it, and reproduced not one term of it. Two reviewers
# reached that independently. It also emitted no HuggingFace front matter, so the Hub rendered the
# page with no licence at all whatever the prose said.
#
# The fix is not more prose. The licence is DATA the publisher supplies, refused rather than
# guessed, the same way every bundled corpus carries its licence as a field.

def test_the_card_refuses_to_assert_a_licence_it_cannot_name(tmp_path, capsys):
    abl = tmp_path / "a.json"
    abl.write_text('{"model": "Qwen/Qwen3-1.7B"}', encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        modelcard.main(["--abliteration", str(abl)])
    message = str(e.value)
    assert "--base-licence is required" in message
    assert "--licence-unknown" in message, "the escape hatch has to be named in the refusal"
    assert "not derivable from its weights" in message, "and why it is not guessed"


def test_the_licence_reaches_the_metadata_the_hub_actually_reads():
    """Prose is not metadata. The Hub renders the badge from the YAML block and nothing else."""
    lines = modelcard.build({"model": "Qwen/Qwen3-1.7B"}, licence="apache-2.0",
                            licence_link="https://www.apache.org/licenses/LICENSE-2.0")
    assert lines[0] == "---"
    head = "\n".join(lines[:lines.index("---", 1)])
    assert "license: apache-2.0" in head
    assert "license_link: https://www.apache.org/licenses/LICENSE-2.0" in head
    assert "- Qwen/Qwen3-1.7B" in head, "base_model is how a reader finds the terms they are under"


def test_the_licence_is_named_in_the_prose_too():
    body = "\n".join(modelcard.build({"model": "Qwen/Qwen3-1.7B"}, licence="apache-2.0"))
    assert "under `apache-2.0`" in body
    assert "The base model is `Qwen/Qwen3-1.7B`" in body


def test_the_card_says_the_weights_were_modified():
    """Apache-2.0 section 4(b) asks a derived work to say so, and several families ask for more."""
    body = "\n".join(modelcard.build({"model": "m"}, licence="apache-2.0"))
    assert "have been modified from the base model" in body
    assert "4(b)" in body


def test_an_unresolved_licence_is_loud_rather_than_absent():
    """Absent reads as "nothing to report", which is a different and flattering claim."""
    body = "\n".join(modelcard.build({"model": "m"}))
    assert "LICENCE UNRESOLVED" in body
    assert "effectively unlicensed" in body
    assert "license: other" in body, "the metadata must not claim a licence either"


def test_nothing_about_the_licence_is_inferred_from_the_model_name():
    """A Llama base does not silently acquire a Llama licence, and must not.

    Guessing would be the same failure as every withdrawn number in this project: a value that
    looks right, is never checked, and is published.
    """
    body = "\n".join(modelcard.build({"model": "meta-llama/Llama-3.2-1B"}))
    assert "LICENCE UNRESOLVED" in body
    assert "llama3" not in body.lower().split("base_model")[0]


# ─────────────────────────────────────────────────────────────────────────────────────
# What the Hub reads, as opposed to what a human reads. Both findings below produce a
# card whose prose is careful and whose metadata publishes the weights as unlicensed.
# ─────────────────────────────────────────────────────────────────────────────────────

def test_an_unresolved_licence_still_renders_on_the_hub():
    """`license: other` with no `license_name` renders nothing at all.

    So `--licence-unknown` produced a card that said UNRESOLVED loudly in its prose and said
    nothing whatever in the block the Hub actually reads.
    """
    block = modelcard.front_matter({"model": "Qwen/Qwen3-1.7B"}, "other", None)
    assert "license: other" in block
    assert any(line.startswith("license_name:") for line in block), (
        "the Hub requires license_name beside `license: other` and renders no badge without it")


def test_a_normal_licence_needs_no_license_name():
    block = modelcard.front_matter({"model": "Qwen/Qwen3-1.7B"}, "apache-2.0", None)
    assert "license: apache-2.0" in block
    assert not any(line.startswith("license_name:") for line in block)


@pytest.mark.parametrize("good", ["apache-2.0", "mit", "gemma", "llama3.2", "cc-by-nc-4.0"])
def test_the_identifiers_the_help_text_names_are_accepted(good):
    assert modelcard.licence_complaint(good) is None


@pytest.mark.parametrize("bad", ["Apache 2.0", "MIT", "Apache-2.0", "apache 2.0"])
def test_a_near_miss_is_refused_rather_than_written_through(bad):
    """The failure is silent on the Hub: it declines the value and shows no licence.

    A publisher who typed the wrong case would have been told nothing by this tool and would
    have found out from a reader, or not at all.
    """
    said = modelcard.licence_complaint(bad)
    assert said and "apache-2.0" in said


def test_the_refusal_reaches_the_shell(tmp_path):
    abl = tmp_path / "abliteration.json"
    abl.write_text('{"model": "Qwen/Qwen3-1.7B", "post_bake_refusals": 0.0}', encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        modelcard.main(["--abliteration", str(abl), "--base-licence", "Apache 2.0"])
    assert "not an identifier the HuggingFace Hub renders" in " ".join(str(e.value).split())


# ── the buyer-facing half: what was measured, and what the page does not cover ──────────
# THE GAP THESE EXIST FOR. The card printed `refusal before: 39.1%` and `after: 1.2%` and nothing
# else: no sample size, no interval, no instrument, none of the five fields that decide whether two
# such numbers describe the same experiment, and not one of the artefact's own sentences saying the
# figure should not be quoted. Every one of those was already in the file the card was reading.

from senbonzakura_check import measurement  # noqa: E402

#: What `cli.Run.eval_provenance` writes, which is where the row count and the disqualifying
#: sentence have been sitting all along.
SELECTION_EVAL = {
    "track": "default", "dataset": "bad_eval_ds", "rows_scored": 131, "partition": "search",
    "search_partition_rows": 131, "held_out": False,
    "note": "These are SELECTION-SET figures: the search chose its winner by scoring these rows.",
}


def _stamped_score(n=128, refusal=0.0234, **over):
    """A `senbonzakura score` artefact, stamped exactly as `score._stamp_refusal` stamps one."""
    res = {"label": "held-out", "model": "edited", "eval": "bad_eval_ds", "n": n,
           "refusal": refusal, "soft_refusal": 0.0, "noncompliant": refusal, "broken": 0.0,
           "heretic": 0.031, "chat_template": "qwen3", "budget_warning": None}
    res.update(over)
    from senbonzakura import metrics as _m

    for estimator, value in (("senbonzakura-ruler", res["refusal"]),
                             ("heretic-keyword", res["heretic"])):
        reported = _m.reportable_rate(round(value * n), n)
        measurement.stamp(res, "refusal_rate", value, estimator, n=n, by_estimator=True,
                          interval=list(reported["ci"]) if reported["ci"] else None,
                          interval_method="Wilson score interval on the refusal count",
                          reportable=reported["reportable"],
                          input_digest="a1b2c3d4e5f60718", partition="measure",
                          prompt_format="qwen3", precision="bfloat16", tool_version="0.4.1")
    return res


class TestWhatWasMeasured:

    def test_the_sample_size_reaches_the_page(self):
        """A rate over nine replies read exactly like a rate over three thousand."""
        text = _text(abl=_abl(refusal_eval=SELECTION_EVAL))
        assert "replies the refusal figures rest on: **131**" in text

    def test_an_unrecorded_sample_says_so_rather_than_being_left_out(self):
        assert modelcard.NOT_RECORDED in _text(abl=_abl())

    def test_every_pinned_field_is_named_even_when_absent(self):
        """The five fields that decide comparability. Silence about them was the old behaviour."""
        text = _text(abl=_abl())
        for _field, label in modelcard.PINNED_LABELS:
            assert label in text, f"{label} is not on the page"

    def test_the_pinned_fields_come_from_the_artefact_that_took_the_measurement(self):
        text = _text(abl=_abl(refusal_eval=SELECTION_EVAL), ref=_stamped_score())
        assert "`a1b2c3d4e5f60718`" in text
        assert "which rows of it: `measure`" in text
        assert "numerical precision: `bfloat16`" in text

    def test_a_separations_partition_is_not_printed_under_a_refusal_rate(self):
        """The subtlety worth a test. An abliteration record stamps the separation statistic,
        whose partition is `search` because the directions are fitted on half the rows. True, and
        true about a different number; the refusal figures' partition is in `refusal_eval`.
        """
        abl = _abl(refusal_eval=SELECTION_EVAL,
                   metrics={"separation": {"metric": "separation", "partition": "search",
                                           "precision": "nf4"}})
        assert modelcard.pinned_fields(abl)["partition"] == "search"
        held = modelcard.pinned_fields(abl, _stamped_score())["partition"]
        assert held == "measure", "the held out figure's own partition was overridden"

    def test_the_estimator_is_read_off_the_stamp_rather_than_assumed(self):
        text = _text(abl=_abl(), ref=_stamped_score())
        assert "`senbonzakura-ruler`" in text and "`heretic-keyword`" in text
        assert "Wilson score interval on the refusal count" in text

    def test_an_unstamped_artefact_says_the_estimator_is_undeclared(self):
        """Hardcoding "this project's ruler" would be a second home for that fact."""
        text = _text(abl=_abl())
        assert "do not declare which estimator produced their figures" in text

    def test_the_interval_is_explained_in_plain_language(self):
        text = _text(abl=_abl())
        assert "How to read the intervals" in text
        assert "have not been shown to differ" in text

    def test_the_generation_budget_reaches_the_page(self):
        text = _text(abl=_abl(generation_settings={"max_new_tokens": 192, "greedy": True}))
        assert "tokens each reply was allowed: **192**" in text
        assert "no sampling noise" in text

    def test_a_card_with_no_artefacts_declares_the_section_unmeasured(self):
        assert modelcard.measured_section() == [modelcard.NOT_MEASURED]

    def test_a_capability_only_card_does_not_describe_refusal_figures_it_has_none_of(self):
        """Six fields of "not recorded" would describe a measurement nobody asked this card for."""
        section = "\n".join(modelcard.measured_section(None, _cap(), None))
        assert "replies the refusal figures rest on" not in section
        assert "Why the rows matter" not in section
        assert "How to read the intervals" in section

    def test_a_stamp_with_no_interval_method_names_the_estimator_anyway(self):
        """An older artefact stamped before intervals were required still gets its instrument named."""
        ref = {"n": 40, "refusal": 0.1}
        measurement.stamp(ref, "refusal_rate", 0.1, "senbonzakura-ruler", n=40, by_estimator=True)
        section = "\n".join(modelcard.measured_section(None, None, ref))
        assert "`senbonzakura-ruler`" in section
        assert "the doubt beside it" not in section

    def test_a_chat_template_recorded_under_either_key_is_read(self):
        assert modelcard.pinned_fields(_abl(chat_template={"name": "qwen3"}))[
            "prompt_format"] == "qwen3"
        assert modelcard.pinned_fields(_abl(chat_template={"source": "tokenizer"}))[
            "prompt_format"] == "tokenizer"
        assert modelcard.pinned_fields(_abl(chat_template="llama3"))["prompt_format"] == "llama3"

    def test_the_readings_come_out_in_a_fixed_order(self):
        """Two runs over one artefact have to produce the same page, byte for byte."""
        ref = _stamped_score()
        assert _text(abl=_abl(), ref=ref) == _text(abl=_abl(), ref=ref)


class TestTheRateCarriesItsCountsAndItsDoubt:

    def test_a_rate_with_a_known_denominator_gets_both(self):
        text = _text(abl=_abl(baseline_refusals=0.391, refusal_eval=SELECTION_EVAL))
        assert "refusal before: **51/131 = 38.9%**" in text and "95% CI" in text

    def test_a_rate_with_no_denominator_says_so_on_the_same_line(self):
        """A caveat under the figure is a caveat that gets quoted away from the figure."""
        line = next(ln for ln in modelcard.refusal_section(_abl(baseline_refusals=0.4))
                    if "refusal before" in ln)
        assert "no interval can be put on it" in line

    def test_an_unreportable_rate_with_no_count_falls_back_rather_than_saying_none_replies(self):
        ref = {"refusal": 0.1}
        measurement.stamp(ref, "refusal_rate", 0.1, "senbonzakura-ruler", by_estimator=True,
                          reportable=False)
        line = "\n".join(modelcard.refusal_section(None, ref))
        assert "None replies" not in line
        assert "no interval can be put on it" in line

    def test_a_sample_too_small_to_carry_a_rate_refuses_to_state_one(self):
        ref = _stamped_score(n=9, refusal=0.1111)
        text = _text(abl=_abl(), ref=ref)
        assert "below the floor this project will state a rate over" in text

    def test_the_heretic_comparable_figure_is_no_longer_dropped(self):
        text = _text(abl=_abl(post_bake_heretic=0.031))
        assert "Heretic's keyword metric" in text

    def test_kl_keeps_its_own_units_and_gains_no_counts(self):
        line = next(ln for ln in modelcard.refusal_section(_abl(post_bake_kl=0.041))
                    if "KL drift" in ln)
        assert line == "- KL drift: **0.041**"

    def test_the_count_is_recovered_exactly_at_the_sizes_this_project_runs(self):
        assert modelcard._count_from(0.0234, 128) == 3
        assert modelcard._count_from(0.391, 131) == 51

    def test_a_sample_beyond_the_recoverable_size_gets_no_invented_numerator(self):
        assert modelcard._count_from(0.5, modelcard.COUNT_RECOVERABLE_N + 1) is None
        assert modelcard._count_from(0.5, 0) is None


class TestTheArtefactsOwnWarningsReachTheReader:

    def test_the_selection_set_note_is_published_beside_the_rate(self):
        """The sharpest of the three. The record says in its own words that the figure is the best
        of however many trials rather than a measurement, and the card dropped the sentence while
        publishing the figure it disqualifies in bold.
        """
        section = "\n".join(modelcard.refusal_section(_abl(refusal_eval=SELECTION_EVAL)))
        assert "SELECTION-SET" in section
        assert "Read the search's own figures with this" in section

    def test_a_budget_warning_travels_with_the_number(self):
        abl = _abl(generation_settings={"max_new_tokens": 48, "greedy": True,
                                        "budget_warning": "the budget is 48 tokens"})
        assert "the budget is 48 tokens" in "\n".join(modelcard.refusal_section(abl))

    def test_the_older_budget_block_spelling_is_read_too(self):
        abl = _abl(reply_budget={"max_new_tokens": 48, "budget_warning": "too short"})
        assert "too short" in "\n".join(modelcard.refusal_section(abl))

    def test_a_self_invalidated_run_says_so(self):
        ref = _stamped_score(self_invalidated="this figure measures the token budget")
        assert any("token budget" in w for _whose, w in modelcard.artefact_warnings(None, ref))

    def test_one_warning_is_not_printed_twice(self):
        ref = _stamped_score(budget_warning="too short",
                             generation_settings={"budget_warning": "too short"})
        texts = [w for _whose, w in modelcard.artefact_warnings(None, ref)]
        assert texts == ["too short"]

    def test_the_selection_note_is_dropped_once_a_held_out_figure_is_supplied(self):
        """Repeating it would warn about a problem the reader has already fixed."""
        warnings = modelcard.artefact_warnings(_abl(refusal_eval=SELECTION_EVAL), _stamped_score())
        assert not any("SELECTION-SET" in w for _whose, w in warnings)

    def test_a_string_where_a_dict_was_expected_is_not_read_as_provenance(self):
        """Older artefacts wrote `refusal_eval` as a sentence. Fail safe, not fail confident."""
        assert modelcard.eval_provenance({"refusal_eval": "rows 0 to 131"}) == {}


class TestTheHeldOutFigureIsTheOneThatMayBePublished:

    def test_the_scored_artefact_supplies_the_headline(self):
        text = _text(abl=_abl(refusal_eval=SELECTION_EVAL), ref=_stamped_score())
        assert "refusal after the edit, by `senbonzakura-ruler`: **3/128 = 2.3%**" in text

    def test_the_searchs_own_figures_are_labelled_as_not_publishable(self):
        text = _text(abl=_abl(baseline_refusals=0.391), ref=_stamped_score())
        assert "not because they are publishable" in text

    def test_the_lecture_and_breakage_rates_come_through_too(self):
        text = _text(abl=_abl(refusal_eval=SELECTION_EVAL), ref=_stamped_score())
        assert "lectured rather than helped" in text and "came out broken" in text

    def test_a_scored_artefact_with_no_figures_in_it_says_so(self):
        text = _text(ref={"n": 40})
        assert "carries no refusal figures" in text

    def test_a_stamp_with_no_interval_falls_back_to_the_rate_and_the_counts(self):
        ref = {"n": 40, "refusal": 0.1}
        measurement.stamp(ref, "refusal_rate", 0.1, "senbonzakura-ruler", n=40, by_estimator=True)
        assert "4/40 = 10.0%" in "\n".join(modelcard.refusal_section(None, ref))

    def test_a_card_built_from_the_scored_artefact_alone_still_renders(self):
        text = _text(ref=_stamped_score())
        assert modelcard.NOT_MEASURED in text          # the abliteration sections
        assert "2.3%" in text                           # and the figure that was supplied


class TestWhatThisDoesNotCover:

    def test_the_section_is_always_there(self):
        for kw in ({}, {"abl": _abl()}, {"abl": _abl(), "cap": _cap()}):
            assert "What this does not cover" in _text(**kw)

    def test_the_standing_limits_are_not_conditional_on_the_figures_looking_good(self):
        text = _text(abl=_abl(), cap=_cap(), ref=_stamped_score())
        for claim in ("Whether the model is safe to deploy", "What a refusal is",
                      "One corpus, one language", "A conversation that continues",
                      "Whether the writing is any good"):
            assert claim in text

    def test_an_unmeasured_capability_is_named_as_a_gap(self):
        assert "What the edit cost" in _text(abl=_abl())
        assert "What the edit cost" not in "\n".join(modelcard.limits_section(_abl(), _cap()))

    def test_a_selection_set_figure_is_named_as_a_gap_with_the_way_out(self):
        text = "\n".join(modelcard.limits_section(_abl(refusal_eval=SELECTION_EVAL)))
        assert "Whether the refusal figures are a measurement" in text
        assert "senbonzakura report --refusal" in text

    def test_an_unestablished_partition_is_its_own_gap(self):
        abl = _abl(refusal_eval=dict(SELECTION_EVAL, held_out=None, note=None))
        assert "Whether the refusal figures were held back" in \
            "\n".join(modelcard.limits_section(abl))

    def test_a_held_out_figure_closes_the_partition_gap(self):
        text = "\n".join(modelcard.limits_section(_abl(refusal_eval=SELECTION_EVAL),
                                                 ref=_stamped_score()))
        assert "Whether the refusal figures are a measurement" not in text

    def test_an_unrecorded_sample_is_named_as_a_gap(self):
        assert "How much evidence is behind the refusal figures" in \
            "\n".join(modelcard.limits_section(_abl()))

    def test_a_sample_under_the_floor_is_named_with_the_floor(self):
        from senbonzakura.metrics import MIN_REPORTABLE_N

        text = "\n".join(modelcard.limits_section(_abl(), ref=_stamped_score(n=9)))
        assert f"fewer than {MIN_REPORTABLE_N}" in text

    def test_a_healthy_sample_raises_no_sample_gap(self):
        text = "\n".join(modelcard.limits_section(_abl(), ref=_stamped_score()))
        assert "Enough replies to state a rate" not in text
        assert "How much evidence is behind" not in text


class TestTheCommandTakesTheScoredArtefact:

    def test_the_scored_artefact_alone_is_enough_to_ask_for_a_card(self, tmp_path, capsys):
        path = tmp_path / "score.json"
        path.write_text(json.dumps(_stamped_score()), encoding="utf-8")
        assert modelcard.main(["--refusal", str(path), "--base-licence", "mit"]) == 0
        assert "2.3%" in capsys.readouterr().out

    def test_the_refusal_flag_is_named_when_nothing_was_supplied(self):
        with pytest.raises(SystemExit) as e:
            modelcard.main(["--base-licence", "mit"])
        assert "--refusal" in str(e.value)

    def test_a_named_scored_artefact_that_is_absent_refuses(self, tmp_path):
        with pytest.raises(SystemExit) as e:
            modelcard.main(["--refusal", str(tmp_path / "nope.json"),
                            "--base-licence", "mit"])
        assert "not a file" in str(e.value)


def test_a_digest_of_several_splits_is_several_facts_not_a_dict_repr():
    text = _text(abl=_abl(track_digest={"bad_ds": "39cc", "good_ds": "6b2b"}))
    assert "`bad_ds` `39cc`, `good_ds` `6b2b`" in text
    assert "{'bad_ds'" not in text


def _tam(**over):
    """A tamper artefact with the fields the section reads, shaped as `tamper` writes them."""
    doc = {
        "bias_direction": "the recovery targets contain the phrases the refusal ruler keys on",
        "refusal_eval": {"partition": "rows-from-128", "n": 392},
        "arms": {
            "recovery": {
                "recipe": {"method": "lora", "steps": 60, "learning_rate": 0.0001,
                           "train_batch": 4, "train_pairs": 128, "train_dtype": "bfloat16",
                           "train_digest": "deadbeef0000",
                           "what_the_method_tests": "an adapter beside the frozen weights"},
                "recovered_fraction": {"point": 0.12, "ci": [0.05, 0.2]},
            },
            "ceiling": {"recovered_fraction": {"point": 0.88, "ci": [0.8, 0.94]}},
        },
        "safety_specific": {"point": 0.1, "ci": [0.03, 0.18],
                            "reading": "the interval excludes zero"},
    }
    doc.update(over)
    return doc


def test_a_card_with_no_tamper_evidence_says_nobody_looked():
    out = "\n".join(modelcard.tamper_section(None))
    assert modelcard.NOT_MEASURED in out
    assert "may be handed back a model that refuses again" in out


def test_a_tamper_run_that_disowned_itself_quotes_no_figure():
    # The artefact still holds numbers when it self-invalidates, and publishing them because they
    # are there is how an invalid measurement becomes a published claim.
    out = "\n".join(modelcard.tamper_section(
        _tam(self_invalidated="the neutral arm's loss rose")))
    assert "DISOWNED ITS OWN FIGURES" in out
    assert "12.0%" not in out
    assert "88.0%" not in out


def test_the_bias_direction_is_stated_before_any_number():
    lines = modelcard.tamper_section(_tam())
    body = "\n".join(lines)
    assert body.index("How to read these") < body.index("12.0%")


def test_the_ceiling_is_reported_so_the_recovery_figure_is_not_read_against_one():
    out = "\n".join(modelcard.tamper_section(_tam()))
    assert "+88.0%" in out
    assert "UNEDITED model" in out


def test_a_recipe_the_artefact_did_not_record_is_said_rather_than_implied():
    doc = _tam()
    doc["arms"]["recovery"].pop("recipe")
    out = "\n".join(modelcard.tamper_section(doc))
    assert "the recipe is not recorded in this artefact" in out


def test_an_injected_field_reads_as_a_sentence_after_a_bold_lead_in():
    out = "\n".join(modelcard.tamper_section(_tam()))
    assert "**How to read these.** The recovery targets" in out


def test_the_limits_bullet_stops_claiming_the_tamper_artefact_is_unread():
    # A disclaimer that outlives the gap it describes is worse than none: a reader believes the
    # disclaimer over the section it contradicts.
    without = "\n".join(modelcard.limits_section())
    assert "`senbonzakura tamper` measure those, and no figure on this page states" in without
    with_tam = "\n".join(modelcard.limits_section(tam=_tam()))
    # The page can read a multiturn artefact since 2026-10-07, so the wording distinguishes one
    # this page cannot read from one that simply was not handed to it. Still a gap either way.
    assert "`senbonzakura multiturn` measures what a continued conversation does" in with_tam
    assert "no figure on this page states it" in with_tam
    assert "What a finetune brings back IS reported above" in with_tam
    assert "`senbonzakura tamper` measure those" not in with_tam


def test_a_tamper_artefact_alone_is_enough_to_build_a_card(tmp_path):
    path = tmp_path / "tamper.json"
    path.write_text(json.dumps(_tam()), encoding="utf-8")
    out = tmp_path / "card.md"
    assert modelcard.main(["--tamper", str(path), "--base-licence", "mit",
                           "--out", str(out)]) == 0
    assert "Tamper resistance" in out.read_text(encoding="utf-8")


def test_the_card_offers_the_tamper_flag_when_it_refuses_for_want_of_artefacts():
    with pytest.raises(SystemExit, match=r"--tamper FILE"):
        modelcard.main(["--base-licence", "mit"])


def test_a_bare_tamper_artefact_reports_what_it_has_and_claims_nothing_it_does_not():
    # Every optional field absent at once: the section must still be a section rather than raising
    # or quietly emitting an empty one, because `tamper` writes these fields conditionally.
    out = "\n".join(modelcard.tamper_section({"arms": {"recovery": {}}}))
    assert "the recipe is not recorded in this artefact" in out
    assert "How to read these" not in out
    assert "UNEDITED model" not in out
    assert "Is it about safety" not in out


def test_a_figure_with_no_interval_is_reported_without_inventing_one():
    doc = _tam()
    doc["arms"]["recovery"]["recovered_fraction"] = {"point": 0.12, "ci": None}
    doc["safety_specific"] = {"point": 0.1, "ci": None}
    out = "\n".join(modelcard.tamper_section(doc))
    assert "**+12.0%** of what there was to recover" in out
    assert "95% CI" not in out


def test_a_safety_gap_with_no_reading_states_the_number_and_stops():
    doc = _tam()
    doc["safety_specific"].pop("reading")
    out = "\n".join(modelcard.tamper_section(doc))
    assert "Is it about safety" in out
    assert "the interval excludes zero" not in out


def test_a_recipe_without_the_method_note_or_a_digest_omits_both_lines():
    doc = _tam()
    doc["arms"]["recovery"]["recipe"] = {"method": "full", "steps": 60}
    out = "\n".join(modelcard.tamper_section(doc))
    assert "method `full`, 60 steps" in out
    assert "what that method does and does not test" not in out
    assert "digest of the training pairs" not in out


@pytest.mark.parametrize(("key", "lead"), [
    ("control_missing", "No control was run."),
    ("ceiling_missing", "No ceiling was run."),
    ("control_caveat", "Read the gap with this."),
    ("budget_warning", "Budget."),
])
def test_each_caveat_the_artefact_carries_reaches_the_page(key, lead):
    out = "\n".join(modelcard.tamper_section(_tam(**{key: "something was wrong with the run"})))
    assert lead in out
    assert "Something was wrong with the run" in out


def test_an_empty_sentence_field_stays_empty_rather_than_raising():
    assert modelcard._sentence("") == ""


# ── the conversation that continues, which the limits section used to only name ──────────────────

def _mt(**over):
    """A multiturn artefact, shaped as `multiturn` writes it.

    The rates and the comparison are built with the real `metrics.reportable_rate` and
    `multiturn.advantage` rather than hand written dicts, so a change in either shape breaks this
    test instead of leaving the card quietly reading a key that is no longer there. That is the
    failure `modelcard._reading` exists for, and it cost this page its two headline numbers once.
    """
    from senbonzakura import metrics
    from senbonzakura import multiturn as multiturn_module

    arm = {"strategy": "ramp", "turns": 5, "refused_at_turn_one": 60,
           "conversion": metrics.reportable_rate(27, 60),
           "first_compliance": {"by_turn": {"1": 0, "2": 9, "3": 11, "4": 5, "5": 2},
                                "median_turn": 3.0, "never": 33}}
    control = {"strategy": "repeat", "turns": 5, "refused_at_turn_one": 60,
               "conversion": metrics.reportable_rate(9, 60),
               "first_compliance": {"by_turn": {"1": 0, "2": 4, "3": 3, "4": 2, "5": 0},
                                    "median_turn": 2.0, "never": 51}}
    doc = {"mode": "multi_turn", "single_turn": {"n": 200}, "refused_at_turn_one": 60,
           "arm": arm, "control": control, "overflowed": 0,
           "advantage": multiturn_module.advantage(arm, control)}
    doc.update(over)
    return doc


def test_a_card_with_no_multiturn_evidence_says_so_rather_than_omitting_the_section():
    out = "\n".join(modelcard.multiturn_section(None))
    assert modelcard.NOT_MEASURED in out
    assert "survives the third" in out


def test_a_multiturn_run_that_disowned_itself_quotes_no_conversion_figure():
    out = "\n".join(modelcard.multiturn_section(
        _mt(self_invalidated="more than a third of the replies were broken")))
    assert "DISOWNED ITS OWN FIGURES" in out
    assert "45.0%" not in out
    assert "+30.0%" not in out


def test_the_control_reading_is_stated_before_the_conversion_figure():
    # Same rule as the bias direction in the tamper section: the quotable number is the one that
    # means nothing on its own, so the reader meets the comparison first.
    body = "\n".join(modelcard.multiturn_section(_mt()))
    assert body.index("How to read these") < body.index("45.0%")


def test_the_control_arm_is_reported_beside_the_escalation():
    body = "\n".join(modelcard.multiturn_section(_mt()))
    assert "27/60 = 45.0%" in body
    assert "9/60 = 15.0%" in body
    assert "asked again" in body


def test_the_denominator_says_the_escalation_ran_only_on_the_refused_rows():
    body = "\n".join(modelcard.multiturn_section(_mt()))
    assert "of 200 attack prompts" in body
    assert "**60** were refused at turn one" in body


def test_an_overlapping_interval_is_not_presented_as_a_difference():
    from senbonzakura import metrics
    from senbonzakura import multiturn as multiturn_module

    doc = _mt()
    doc["control"]["conversion"] = metrics.reportable_rate(24, 60)
    doc["advantage"] = multiturn_module.advantage(doc["arm"], doc["control"])
    body = "\n".join(modelcard.multiturn_section(doc))
    assert "THE TWO INTERVALS OVERLAP" in body
    assert "do not credit the strategy with the difference" in body


def test_a_sample_below_the_floor_gives_counts_rather_than_a_rate():
    from senbonzakura import metrics

    doc = _mt()
    doc["arm"]["conversion"] = metrics.reportable_rate(2, 4)
    doc["advantage"] = None
    body = "\n".join(modelcard.multiturn_section(doc))
    assert "**2/4**" in body
    assert "50.0%" not in body


def test_the_rows_that_hit_the_budget_are_disclosed_rather_than_counted_as_refusals():
    body = "\n".join(modelcard.multiturn_section(_mt(overflowed=7)))
    assert "**7 rows hit the token budget**" in body
    assert "unknown rather than a refusal" in body


def test_a_missing_control_is_disclosed_on_the_page():
    body = "\n".join(modelcard.multiturn_section(
        _mt(control=None, advantage=None,
            control_missing="the plain-repetition control was not run")))
    assert "No control was run." in body
    assert "The plain-repetition control was not run" in body


def test_the_median_turn_reads_as_a_turn_rather_than_a_float():
    body = "\n".join(modelcard.multiturn_section(_mt()))
    assert "median was turn **3**" in body


@pytest.mark.parametrize(("tam", "mt", "expected", "forbidden"), [
    (None, None, "no figure on this page states either", "IS reported above"),
    ("yes", None, "What a finetune brings back IS reported above", "states either"),
    (None, "yes", "What a continued conversation does IS reported above", "states either"),
    ("yes", "yes", "Both are measured in their own sections above", "no figure on this page"),
])
def test_the_single_request_caveat_says_only_what_is_still_true(tam, mt, expected, forbidden):
    """A disclaimer that outlives its gap is worse than none: it tells a reader the page is
    silent on something the page now reports, and they believe the disclaimer over the section.
    """
    bullet = modelcard._single_request_bullet(tam, mt)
    assert expected in bullet
    assert forbidden not in bullet


def test_the_limits_section_stops_claiming_the_multiturn_artefact_is_unread():
    out = "\n".join(modelcard.limits_section(tam=_tam(), mt=_mt()))
    assert "not read by this page" not in out
    assert "no figure on this page states" not in out


def test_a_multiturn_artefact_alone_is_enough_to_build_a_card(tmp_path, capsys):
    path = tmp_path / "multiturn.json"
    path.write_text(json.dumps(_mt()), encoding="utf-8")
    assert modelcard.main(["--multiturn", str(path), "--base-licence", "mit"]) == 0
    out = capsys.readouterr().out
    assert "A conversation that continues, which is whether it holds" in out
    assert "27/60 = 45.0%" in out


def test_the_usage_refusal_names_the_multiturn_flag_among_the_artefacts():
    with pytest.raises(SystemExit, match=r"--multiturn FILE"):
        modelcard.main(["--base-licence", "mit"])


@pytest.mark.parametrize("which", ["tam", "mt"])
def test_a_run_that_disowned_itself_is_still_a_gap_in_the_limits_section(which):
    """Found by rendering the real multiturn smoke artefact: it self-invalidated, the section
    quoted no figure, and the caveat said the page reported it anyway.
    """
    doc = (_tam() if which == "tam" else _mt())
    doc["self_invalidated"] = "fewer than 30 prompts refused at turn one"
    out = "\n".join(modelcard.limits_section(**{which: doc}))
    assert "IS reported above" not in out
    assert "no figure on this page states either" in out


def test_a_saturated_ceiling_is_stated_before_the_recovery_figure():
    """A saturated ceiling changes what the figure means rather than qualifying it, so it cannot
    sit in the caveat list under the number it disarms.
    """
    body = "\n".join(modelcard.tamper_section(
        _tam(ceiling_saturated="the recipe refused every prompt on the unedited model")))
    assert "THE DOSE SATURATED" in body
    assert body.index("THE DOSE SATURATED") < body.index("12.0%")
    assert "The recipe refused every prompt on the unedited model" in body


def test_a_card_without_saturation_does_not_mention_the_dose():
    assert "THE DOSE SATURATED" not in "\n".join(modelcard.tamper_section(_tam()))
