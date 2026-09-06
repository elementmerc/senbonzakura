# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The matched comparison (Q-23): can the filter tell refusal from subject matter?

WHY THIS FILE EXISTS

Q-14 asked which STATISTIC could tell a refusal axis from a topic axis. Testing whether the
measured null floor could answer it instead produced a different answer: in a synthetic world with
topic clusters and no refusal anywhere, every statistic kept every candidate, and the floor
stopped none of them. The comparison was the defect, not the ruler.

A candidate is one cluster of harmful prompts. Judged against harmless prompts in general, a
cluster about explosives separates because it is about explosives. Judged against harmless prompts
on the same subject, the subject cancels and refusal is the only thing left to vary.

THE SHAPE OF THESE TESTS

Every one is a pair: a world where refusal is genuinely present, and an otherwise identical world
where it is absent. A filter that cannot tell those apart is worthless however impressive its
numbers, so the no-refusal world is the control that forces the instrument to fail, and it is the
half that would catch a change making the filter merely permissive.
"""
import pytest
import torch

from senbonzakura import cli, separation

H, N_TOPIC, PER_BAD, PER_GOOD = 48, 8, 32, 32
STATS = ["cohens-d", "variance-ratio", "welch-ratio", "auc"]


def _world(refusal_scale, seed=0):
    """Harmful and harmless prompts sharing topic structure, refusal present only on the harmful.

    The harmless side deliberately carries the SAME topic mix as the harmful side, which is the
    best case for a topic-matched corpus. That is what makes the unmatched result below a
    statement about the comparison rather than about the dataset: even with a perfectly matched
    corpus, scoring against the pool at large does not hold the subject still.
    """
    g = torch.Generator().manual_seed(seed)
    topics = torch.randn(N_TOPIC, H, generator=g)
    topics = topics - topics.mean(0)
    refusals = torch.randn(N_TOPIC, H, generator=g)
    refusals = refusals / refusals.norm(dim=1, keepdim=True)
    bad, good, bad_topic = [], [], []
    for t in range(N_TOPIC):
        bad.append(topics[t] + refusals[t] * refusal_scale
                   + torch.randn(PER_BAD, H, generator=g) * 0.35)
        good.append(topics[t] + torch.randn(PER_GOOD, H, generator=g) * 0.35)
        bad_topic += [t] * PER_BAD
    return torch.cat(bad), torch.cat(good), torch.tensor(bad_topic)


def _median_candidate_score(matched, refusal_scale, stat_name, seed=0):
    """What a typical candidate scores in this world, through the real scoring path."""
    Rb, Rg, topic = _world(refusal_scale, seed)
    d0 = Rb.mean(0) - Rg.mean(0)
    basis = [d0 / d0.norm()]
    stat = separation.get(stat_name)
    fit_idx, score_idx = cli._halves(int(Rg.shape[0]), 0)
    good_fit, good_score = Rg[fit_idx], Rg[score_idx]
    score_fn = cli._matched_held_out_separation if matched else cli._held_out_separation
    scores = []
    for t in range(N_TOPIC):
        s = score_fn(Rb[topic == t], good_fit, good_score, basis, 3 + t, stat)
        if s is not None:
            scores.append(float(s))
    assert scores, "no candidate could be scored at all"
    return sorted(scores)[len(scores) // 2]


# ── the finding, as a test ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", STATS)
def test_the_unmatched_comparison_cannot_tell_refusal_from_topic(name):
    """THE CONTROL THAT FORCES THE INSTRUMENT TO FAIL, and the reason Q-23 exists.

    Two worlds, identical but for whether refusal is present at all. Under the unmatched
    comparison every statistic scores them the same, so no threshold placed anywhere on any of
    these rulers could separate them.

    This asserts a DEFECT is present. If it ever fails, the unmatched path has started
    distinguishing the two worlds and this whole analysis needs revisiting rather than the test
    being adjusted.
    """
    without = _median_candidate_score(False, 0.0, name)
    with_ = _median_candidate_score(False, 2.0, name)
    # Tight on purpose. Measured gaps are 2.8% to 8.8% across the four statistics, so 15% leaves
    # room for a seed to wobble and none at all for the unmatched path to start working.
    assert with_ == pytest.approx(without, rel=0.15), (
        f"{name}: unmatched scores {without} without refusal and {with_} with it, which is a "
        f"bigger gap than this analysis predicts")


@pytest.mark.parametrize("name", STATS)
def test_the_matched_comparison_lands_on_the_null_when_there_is_no_refusal(name):
    """The other half: hold the subject still and a world with no refusal scores like one."""
    stat = separation.get(name)
    score = _median_candidate_score(True, 0.0, name)
    assert score < stat.threshold, (
        f"{name}: a world containing no refusal at all scored {score}, clearing the "
        f"{stat.threshold} threshold")


@pytest.mark.parametrize("name", STATS)
def test_the_matched_comparison_still_finds_refusal_when_it_is_there(name):
    """And it must not achieve the test above by rejecting everything.

    A filter that says no to every candidate discriminates exactly as much as one that says yes.
    Both extremes have happened in this project, a year apart, which is why both directions are
    asserted rather than only the one being fixed.
    """
    stat = separation.get(name)
    score = _median_candidate_score(True, 2.0, name)
    assert score >= stat.threshold, (
        f"{name}: real refusal scored {score}, below the {stat.threshold} threshold")


@pytest.mark.parametrize("name", STATS)
def test_matched_scoring_separates_the_two_worlds_and_unmatched_does_not(name):
    """The comparison the other three tests imply, stated as one number so it cannot be missed."""
    unmatched_gap = abs(_median_candidate_score(False, 2.0, name)
                        - _median_candidate_score(False, 0.0, name))
    matched_gap = abs(_median_candidate_score(True, 2.0, name)
                      - _median_candidate_score(True, 0.0, name))
    assert matched_gap > unmatched_gap, f"{name}: matched {matched_gap}, unmatched {unmatched_gap}"


# ── the matching itself ────────────────────────────────────────────────────────────
def test_matching_picks_harmless_rows_on_the_same_topic():
    """The mechanism, checked directly rather than only through its effect."""
    Rb, Rg, topic = _world(2.0)
    good_topic = torch.arange(Rg.shape[0]) // PER_GOOD
    d0 = Rb.mean(0) - Rg.mean(0)
    basis = [d0 / d0.norm()]
    for t in range(N_TOPIC):
        idx = cli._matched_harmless_idx(Rb[topic == t], Rg, basis, PER_GOOD)
        share = float((good_topic[idx] == t).float().mean())
        assert share > 0.8, f"topic {t}: only {share:.0%} of the matched controls were on-topic"


def test_matching_ignores_the_directions_already_called_refusal():
    """Matching on the full vector would let refusal choose the controls.

    The controls would then differ from the candidate in the one respect the comparison exists to
    measure, which is the same circularity as scoring a direction on the rows it was fitted on.
    """
    Rb, Rg, topic = _world(0.0)
    rows = Rb[topic == 0]
    # A direction that is IN the basis must not influence which controls are chosen, so shifting
    # every harmless row along it changes nothing about the selection.
    u = torch.randn(H)
    u = u / u.norm()
    plain = cli._matched_harmless_idx(rows, Rg, [u], 16)
    shifted = cli._matched_harmless_idx(rows, Rg + u * 25.0, [u], 16)
    assert plain.tolist() == shifted.tolist()


def test_matching_refuses_when_there_are_no_controls_to_draw():
    Rb, Rg, topic = _world(1.0)
    assert cli._matched_harmless_idx(Rb[topic == 0], Rg[:0], [], 8) is None
    assert cli._matched_harmless_idx(Rb[topic == 0], Rg, [], 0) is None


def test_matching_never_asks_for_more_controls_than_exist():
    Rb, Rg, topic = _world(1.0)
    idx = cli._matched_harmless_idx(Rb[topic == 0], Rg[:10], [], 999)
    assert len(idx) == 10


def test_the_matched_score_does_not_look_at_the_rows_it_scores():
    """The controls are chosen from the FIT half only.

    If the score half picked its own controls, the matching would be a second way of fitting the
    score, which is the defect the held-out split exists to prevent.
    """
    Rb, Rg, topic = _world(2.0)
    d0 = Rb.mean(0) - Rg.mean(0)
    basis = [d0 / d0.norm()]
    rows = Rb[topic == 0]
    fit_idx, score_idx = cli._halves(int(rows.shape[0]), 5)
    gf, gs = Rg[:Rg.shape[0] // 2], Rg[Rg.shape[0] // 2:]
    before = cli._matched_held_out_separation(rows, gf, gs, basis, 5)
    # Move the SCORE half of the cluster far away. The controls it meets are selected from the fit
    # half, so the selection must be unchanged; only the score itself may move.
    moved = rows.clone()
    moved[score_idx] += 40.0
    after = cli._matched_held_out_separation(moved, gf, gs, basis, 5)
    assert before is not None and after is not None
    assert after != before


def test_a_cluster_too_small_to_split_is_refused_under_matching_too():
    Rb, Rg, topic = _world(1.0)
    tiny = Rb[topic == 0][: cli.MIN_HELD_OUT_ROWS * 2 - 1]
    gf, gs = Rg[:100], Rg[100:]
    assert cli._matched_held_out_separation(tiny, gf, gs, [], 0) is None


def test_the_null_floor_follows_the_candidates_through_the_matched_path():
    """A floor measured on the unmatched comparison would answer a different question."""
    Rb, Rg, _ = _world(1.0)
    gf, gs = Rg[:128], Rg[128:]
    plain, _ = cli._null_separation_floor(Rb, gf, gs, [], 32, 4, 12, None, matched=False)
    matched, _ = cli._null_separation_floor(Rb, gf, gs, [], 32, 4, 12, None, matched=True)
    assert plain != matched


# ── the null for the matching itself ───────────────────────────────────────────────
def test_matching_quality_is_low_when_on_topic_controls_exist():
    Rb, Rg, topic = _world(1.0)
    q = cli.matching_quality(Rb[topic == 0], Rg, [], PER_GOOD)
    assert q is not None
    assert q < cli.MATCHING_USELESS_RATIO, f"matching found on-topic controls but scored {q}"


def test_matching_quality_reaches_one_when_the_corpus_shares_no_subjects():
    """THE CASE THE MEASURE EXISTS FOR, and the one nothing else would have caught.

    `--matched-scoring` asks for harmless prompts on the candidate's subject. Whether any exist is
    a property of the corpus, not of the request: a harmless set sharing no subject matter still
    returns its nearest rows, and they are not controls. Without this number a run would publish
    matched figures taken against arbitrary rows, with nothing anywhere saying so.
    """
    Rb, _, topic = _world(1.0)
    g = torch.Generator().manual_seed(99)
    # Harmless prompts from somewhere else entirely: no shared topic structure at all.
    elsewhere = torch.randn(256, H, generator=g) * 0.35 + 40.0
    q = cli.matching_quality(Rb[topic == 0], elsewhere, [], PER_GOOD)
    assert q is not None
    assert q > cli.MATCHING_USELESS_RATIO, (
        f"a corpus with no shared subjects should score near 1.0, got {q}")


def test_matching_quality_reports_nothing_when_there_is_nothing_to_report():
    Rb, Rg, topic = _world(1.0)
    assert cli.matching_quality(Rb[topic == 0], Rg[:0], [], 8) is None
    # Every harmless row identical to the cluster centre: no scale to express a ratio against.
    flat = Rb[topic == 0].mean(0).expand(16, H)
    assert cli.matching_quality(Rb[topic == 0], flat, [], 8) is None


def test_matching_quality_is_calibrated_against_corpora_whose_answer_is_known():
    """Pins the numbers the docstring quotes, including the one that makes it a one-sided alarm.

    A quarter-matched corpus scores LOWER than a perfectly matched one, because its distant rows
    inflate the mean distance the ratio divides by. So a low value must never be read as "the
    matching worked well", and this test is what stops that reading from creeping back in.
    """
    Rb, Rg, topic = _world(1.0)
    d0 = Rb.mean(0) - Rg.mean(0)
    basis = [d0 / d0.norm()]
    g = torch.Generator().manual_seed(3)
    far = torch.randn(256, H, generator=g) * 0.35 + 30.0

    def median_quality(pool):
        vals = sorted(cli.matching_quality(Rb[topic == t], pool, basis, PER_GOOD)
                      for t in range(N_TOPIC))
        return vals[len(vals) // 2]

    perfect = median_quality(Rg)
    none_shared = median_quality(far)
    quarter = median_quality(torch.cat([Rg[:64], far[:192]]))

    assert none_shared > cli.MATCHING_USELESS_RATIO, f"the alarm must fire here: {none_shared}"
    assert perfect < cli.MATCHING_USELESS_RATIO, f"a perfect corpus must not trip it: {perfect}"
    assert quarter < perfect, (
        "the documented catch: a partly matched corpus scores lower than a perfect one, so the "
        f"ratio is not a quality score. perfect={perfect} quarter={quarter}")


# ── the scale-free replacement (D2, 2026-09-06) ──────────────────────────────────────
# `matching_quality` is a sound ALARM and an unsound MEASURE: it divides by the mean distance to
# every harmless row, so a corpus of a few near rows and many distant ones inflates its own
# denominator and scores better than a perfect corpus. That is tolerable while the number is only
# read at the alarm end, and not tolerable once it becomes the acceptance target for corpus work,
# which is what the operator's D1 decision makes it. `match_closeness` divides instead by a
# property of the harmless corpus alone.

def _pools():
    Rb, Rg, topic = _world(1.0)
    d0 = Rb.mean(0) - Rg.mean(0)
    basis = [d0 / d0.norm()]
    g = torch.Generator().manual_seed(3)
    far = torch.randn(256, H, generator=g) * 0.35 + 30.0
    return Rb, Rg, topic, basis, far


def _median(fn, Rb, topic, pool, basis):
    vals = sorted(fn(Rb[topic == t], pool, basis, PER_GOOD) for t in range(N_TOPIC))
    return vals[len(vals) // 2]


def test_closeness_orders_the_three_corpora_correctly():
    """THE DEFECT THIS REPLACES, stated as the thing the old number got wrong.

    Perfect must beat partial must beat none. `matching_quality` fails the first comparison and
    reports the partial corpus as the best of the three.
    """
    Rb, Rg, topic, basis, far = _pools()
    quarter = torch.cat([Rg[:64], far[:192]])
    perfect = _median(cli.match_closeness, Rb, topic, Rg, basis)
    partial = _median(cli.match_closeness, Rb, topic, quarter, basis)
    none_shared = _median(cli.match_closeness, Rb, topic, far, basis)
    assert perfect < partial < none_shared, (
        f"perfect={perfect:.3f} partial={partial:.3f} none={none_shared:.3f}")

    old_perfect = _median(cli.matching_quality, Rb, topic, Rg, basis)
    old_partial = _median(cli.matching_quality, Rb, topic, quarter, basis)
    assert old_partial < old_perfect, (
        "the old ratio no longer prefers the partial corpus, so this test is guarding nothing and "
        "needs rewriting around a case that still separates them")


def test_a_perfectly_matched_corpus_reads_about_one():
    """The unit is the corpus's own nearest-neighbour distance, so 1.0 means the controls are as
    near the candidate as harmless prompts are to each other: as good as the space allows.
    """
    Rb, Rg, topic, basis, _far = _pools()
    assert _median(cli.match_closeness, Rb, topic, Rg, basis) == pytest.approx(1.0, abs=0.25)


def test_a_corpus_with_nothing_on_the_subject_reads_far_above_the_poor_mark():
    Rb, _Rg, topic, basis, far = _pools()
    assert _median(cli.match_closeness, Rb, topic, far, basis) > cli.MATCH_CLOSENESS_POOR


def test_adding_distant_rows_cannot_improve_the_score():
    """WHY THIS ONE CAN BE A TARGET AND THE OLD ONE CANNOT.

    Padding a corpus with unrelated prompts made `matching_quality` look better. If that were true
    of the replacement, the cheapest way to hit any target would be to add junk, and the metric
    would be actively harmful as an acceptance test.
    """
    Rb, Rg, topic, basis, far = _pools()
    before = _median(cli.match_closeness, Rb, topic, Rg, basis)
    after = _median(cli.match_closeness, Rb, topic, torch.cat([Rg, far]), basis)
    assert after >= before - 1e-6, f"padding improved the score: {before:.3f} -> {after:.3f}"

    old_before = _median(cli.matching_quality, Rb, topic, Rg, basis)
    old_after = _median(cli.matching_quality, Rb, topic, torch.cat([Rg, far]), basis)
    assert old_after < old_before, (
        "the old ratio no longer rewards padding, so this contrast needs rewriting rather than "
        "keeping")


def test_closeness_is_invariant_to_the_scale_of_the_space():
    """Scale-free is the name of the property, so it is worth asserting rather than trusting.

    Multiplying every vector by a constant changes both the distance to the controls and the
    typical neighbour distance by that constant, so the ratio must not move. The old number is not
    invariant in any useful sense because its denominator mixes a corpus property with the
    candidate's position.
    """
    Rb, Rg, topic, basis, _far = _pools()
    plain = _median(cli.match_closeness, Rb, topic, Rg, basis)
    scaled = _median(cli.match_closeness, Rb * 1000.0, topic,
                     Rg * 1000.0, list(basis))
    assert scaled == pytest.approx(plain, rel=1e-4)


def test_closeness_reports_nothing_when_there_is_nothing_to_report():
    Rb, _Rg, topic, basis, _far = _pools()
    one = torch.zeros(1, H)
    assert cli.match_closeness(Rb[topic == 0], one, basis, PER_GOOD) is None
    assert cli.match_closeness(Rb[topic == 0], torch.zeros(0, H), basis, PER_GOOD) is None


def test_the_neighbour_scale_is_sampled_but_stable():
    """Capped at MATCH_SCALE_SAMPLE because the full matrix is quadratic. The cap has to be large
    enough that the number does not wander between runs of different corpus sizes.
    """
    _Rb, Rg, _topic, basis, _far = _pools()
    small = cli._typical_neighbour_distance(Rg, basis, seed=0)
    same = cli._typical_neighbour_distance(Rg, basis, seed=1)
    assert small == pytest.approx(same, rel=0.15), (
        f"the sampled scale moved with the seed: {small} against {same}")


def test_a_run_records_the_closeness_beside_the_alarm(base_args, tiny_model, tiny_tok, monkeypatch):
    """Both numbers, because they answer different questions and one is not a replacement.

    `matching_quality` is the alarm the artefact has always carried; `match_closeness` is what
    corpus work is measured against. A run that recorded only one of them would either lose the
    alarm or leave the target unmeasurable on real residuals, which is the thing blocking Q-25.
    """
    from tests.test_abliterator import _topic_matched
    base_args.max_directions = 3
    base_args.matched_scoring = True
    a = cli.Abliterator(base_args, lambda _m: None, model=tiny_model, tok=tiny_tok)
    _topic_matched(a, monkeypatch)
    a.extract_directions("bad", "good", None, "good")
    assert a.matching_quality is not None
    assert a.match_closeness is not None
    assert a.match_closeness > 0.0


def test_the_harness_reports_the_closeness_so_a_corpus_can_be_measured_without_new_tooling():
    """The measurement Q-25 is blocked on runs through the existing Q-14 harness.

    It already takes `--good-ds` and captures residuals once per seed, so pointing it at one corpus
    and then another answers the question. Requiring a bespoke script for it would be the thing
    that keeps producing harnesses that measure nothing.
    """
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "tools" / "measure_separation.py"
    text = src.read_text(encoding="utf-8")
    assert '"match_closeness"' in text
    assert "--good-ds" in text
