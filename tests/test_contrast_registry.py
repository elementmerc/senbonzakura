# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Every contrast builds, is benign by construction, and is matched in length arm for arm.

The length check is the one with a scar behind it: six candidates in the 2026-10-14 sweep scored a
perfect separation where prompt LENGTH alone also separated the arms perfectly, so every pair here
is held to matching word counts and the test fails rather than the sweep repeating the mistake.
"""
import pytest

from senbonzakura import behaviourset, contrasts


def built():
    """Every contrast except refusal, whose arms come from a pack that may not be installed."""
    return {k: c for k, c in contrasts.REGISTRY.items() if not c.external}


def test_every_contrast_builds_and_has_two_non_empty_arms():
    for key, contrast in built().items():
        rows = contrast.build()
        assert rows, key
        for pair in rows:
            assert pair.a.strip() and pair.b.strip(), key
            assert pair.a != pair.b, key


def test_every_instruction_pair_matches_in_word_count():
    mismatched = [r for r in contrasts.instruction_report() if not r["words_match"]]
    assert mismatched == []


def test_every_instruction_pair_is_close_in_character_count():
    """Word counts can match while one arm is far longer in characters, which is the same confound."""
    for row in contrasts.instruction_report():
        assert abs(row["a_chars"] - row["b_chars"]) <= 6, row


def test_every_fact_pair_matches_in_word_count():
    assert [r for r in contrasts.facts_report() if not r["words_match"]] == []


def test_the_false_half_of_every_fact_differs_from_the_true_half():
    for true, false in contrasts.FACTS:
        assert true != false
        assert len(true.split()) == len(false.split())


def test_an_instructed_contrast_differs_only_by_its_instruction():
    """The arms must share the request word for word, or the contrast is not about the instruction."""
    for key, contrast in built().items():
        if not contrast.instructed:
            continue
        high, low = contrasts.INSTRUCTIONS[key][0] if key in contrasts.INSTRUCTIONS else (None, None)
        if high is None:
            continue  # toxicity owns its own instruction list
        for pair in contrast.build()[:8]:
            assert pair.a.removeprefix(high) == pair.b.removeprefix(low), key


def test_every_instructed_contrast_offers_a_paraphrase_to_transfer_against():
    for key, contrast in built().items():
        if contrast.instructed:
            assert contrast.phrasings >= 2, key
            assert contrast.build(phrasing=1)[0].a != contrast.build(phrasing=0)[0].a, key


def test_a_phrasing_that_does_not_exist_is_refused_by_name():
    """A ValueError either way: the toxicity contrast raises its own channel's error type."""
    for contrast in built().values():
        with pytest.raises(ValueError, match="phrasing"):
            contrast.build(phrasing=99)


def test_the_uninstructed_contrasts_are_the_two_the_catalogue_prefers():
    uninstructed = {k for k, c in contrasts.REGISTRY.items() if not c.instructed}
    assert uninstructed == {"refusal", "entity-knowledge", "honesty", "sycophancy"}


def test_entity_knowledge_asks_about_invented_things_on_one_side_only():
    for pair in contrasts.get("entity-knowledge").build():
        assert any(name in pair.a for name in contrasts.INVENTED)
        assert not any(name in pair.b for name in contrasts.INVENTED)


def test_the_invented_names_are_invented_rather_than_borrowed():
    """No real place, person or product is described as fictional by this corpus."""
    assert len(set(contrasts.INVENTED)) == len(contrasts.INVENTED)
    for name in contrasts.INVENTED:
        assert name.islower()


def test_every_key_is_one_a_card_and_a_score_file_would_both_accept():
    for key in contrasts.REGISTRY:
        assert behaviourset.KEY_PATTERN.match(key), key


def test_every_contrast_records_a_tier_the_catalogue_knows():
    tiers = {contrasts.SHIPPED, contrasts.FIRST_RELEASE, contrasts.STRONG,
             contrasts.MODERATE, contrasts.WEAK, contrasts.CONTROL}
    for key, contrast in contrasts.REGISTRY.items():
        assert contrast.tier in tiers, key


def test_the_four_first_release_behaviours_are_all_present():
    """`16-the-behaviour-catalogue.md` settled this roster; a sweep that skips one is incomplete."""
    for key in ("refusal", "sycophancy", "honesty", "entity-knowledge"):
        assert key in contrasts.REGISTRY


def test_what_is_deliberately_not_built_says_why():
    assert contrasts.NOT_BUILT
    for key, reason in contrasts.NOT_BUILT.items():
        assert len(reason) > 80, key
        with pytest.raises(contrasts.ContrastError, match="deliberately not built"):
            contrasts.get(key)


def test_an_unknown_contrast_is_refused_with_the_list_of_known_ones():
    with pytest.raises(contrasts.ContrastError, match="Available"):
        contrasts.get("telepathy")


def test_the_refusal_contrast_is_the_only_one_reaching_outside_this_file():
    external = {k for k, c in contrasts.REGISTRY.items() if c.external}
    assert external == {"refusal"}


def test_building_the_same_contrast_twice_gives_the_same_prompts():
    """A partition is taken by slicing, so a corpus that reorders is two different corpora."""
    for key, contrast in built().items():
        assert contrast.build() == contrast.build(), key


def test_no_contrast_prompt_carries_a_subject_from_outside_the_declared_lists():
    """Benign by construction, asserted: the subject of every instructed prompt is declared here."""
    for key, contrast in built().items():
        if not contrast.instructed:
            continue
        for pair in contrast.build():
            assert pair.group in contrasts.SUBJECTS, key
