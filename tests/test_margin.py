"""Tests for the criterion-free compass (senbonzakura.margin).

Generation three of this measurement has no prose to parse, which removes the bug
class that produced both of the scoring errors in this project's history. What is
left to get wrong is arithmetic and indexing, so that is what these cover: which
token ids get read, which *position* they get read at, and whether the ranking
statistic means what the docstring says.

The module had no tests at all before this file, and it is the module every
published AUC comes from.
"""
import json
import random
from pathlib import Path

import pytest
import torch

from senbonzakura import margin


@pytest.fixture
def margin_kit(model_factory, tok_factory):
    """A model and tokenizer whose HARMFUL and BENIGN id sets do not overlap.

    The fixture tokenizer's default 16-token vocabulary maps one spelling of each
    word onto the same id, which would attenuate every margin toward zero and make
    a real discrimination failure look like a fixture quirk. 32 is the smallest
    clean width.
    """
    return model_factory(H=8, NL=4, V=32), tok_factory(vocab_size=32)


# ── label_token_ids ────────────────────────────────────────────────────────────────
def test_label_token_ids_collects_every_spelling(margin_kit):
    _, tok = margin_kit
    ids = margin.label_token_ids(tok, "HARMFUL")
    assert ids == sorted(set(ids))          # sorted and deduplicated
    assert len(ids) > 1                     # the spellings genuinely differ
    assert all(0 <= i < 32 for i in ids)


def test_label_token_ids_of_the_two_verdicts_do_not_overlap(margin_kit):
    """Overlapping ids would put the same logit on both sides of the subtraction."""
    _, tok = margin_kit
    assert not set(margin.label_token_ids(tok, "HARMFUL")) & set(margin.label_token_ids(tok, "BENIGN"))


def test_label_token_ids_survives_a_spelling_that_encodes_to_nothing(margin_kit):
    """A variant the tokenizer maps to no tokens is skipped, not indexed into."""
    _, tok = margin_kit
    assert margin.label_token_ids(tok, "") == []


# ── auc ────────────────────────────────────────────────────────────────────────────
def test_auc_perfect_separation():
    assert margin.auc([3.0, 2.0], [1.0, 0.0]) == 1.0


def test_auc_perfect_inversion():
    assert margin.auc([0.0, 1.0], [2.0, 3.0]) == 0.0


def test_auc_ties_count_half():
    """A saturated model gives identical margins everywhere; that is chance, not skill."""
    assert margin.auc([1.0, 1.0], [1.0, 1.0]) == 0.5


def test_auc_hand_computed_mixture():
    # pos 2.0 beats both negatives (2 wins); pos 0.0 loses to both (0). 2 / 4 pairs.
    assert margin.auc([2.0, 0.0], [1.0, 1.0]) == 0.5


def test_auc_counts_a_tie_against_a_win_correctly():
    # 1.0 vs [1.0, 0.0]: one tie (0.5) and one win (1.0) out of 2 pairs.
    assert margin.auc([1.0], [1.0, 0.0]) == 0.75


@pytest.mark.parametrize(("pos", "neg"), [([], [1.0]), ([1.0], []), ([], [])])
def test_auc_of_an_empty_arm_is_none_not_a_number(pos, neg):
    """None is the honest answer, and main() is required to refuse it (see below)."""
    assert margin.auc(pos, neg) is None


def test_auc_of_scrambled_labels_lands_at_chance():
    """The control that says the statistic is not manufacturing separation.

    Random labels over 200 against 200 give AUC ~ N(0.5, 0.029): the standard
    error is sqrt((n + m + 1) / (12nm)) = sqrt(401 / 480000). The tolerance is
    four of those, deliberately loose. Tightening it towards one SE turns a
    correct test into one that fails about a third of the time.
    """
    rng = random.Random(20260729)
    pool = [rng.gauss(0.0, 1.0) for _ in range(400)]
    rng.shuffle(pool)
    assert abs(margin.auc(pool[:200], pool[200:]) - 0.5) < 4 * 0.029


def _naive_auc(pos, neg):
    """The definition, written out: the reference the fast rank form must match."""
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_the_rank_form_equals_the_pairwise_definition(seed):
    """auc() was rewritten from O(n*m) to ranks for the bootstrap; it must not have moved.

    Values are drawn from a small integer range on purpose, so ties are frequent
    rather than rare: ties are where a rank implementation goes wrong.
    """
    rng = random.Random(seed)
    pos = [float(rng.randint(0, 4)) for _ in range(40)]
    neg = [float(rng.randint(0, 4)) for _ in range(35)]
    assert margin.auc(pos, neg) == pytest.approx(_naive_auc(pos, neg), abs=1e-12)


def test_the_rank_form_handles_an_all_ties_arm():
    assert margin.auc([2.0] * 10, [2.0] * 10) == pytest.approx(_naive_auc([2.0] * 10, [2.0] * 10))


# ── bootstrap intervals (task 9) ───────────────────────────────────────────────────
def test_the_interval_brackets_the_point_estimate():
    rng = random.Random(7)
    pos = [rng.gauss(1.0, 1.0) for _ in range(120)]
    neg = [rng.gauss(0.0, 1.0) for _ in range(120)]
    lo, hi = margin.bootstrap_auc_ci(pos, neg, seed=42, resamples=400)
    assert lo < margin.auc(pos, neg) < hi


def test_the_interval_is_reproducible_from_the_seed():
    pos, neg = [3.0, 2.0, 1.5, 4.0], [1.0, 0.5, 2.5, 0.0]
    first = margin.bootstrap_auc_ci(pos, neg, seed=99, resamples=200)
    assert first == margin.bootstrap_auc_ci(pos, neg, seed=99, resamples=200)
    assert first != margin.bootstrap_auc_ci(pos, neg, seed=100, resamples=200)


def test_a_wider_interval_for_fewer_prompts():
    """The reason the interval exists: n=200 and n=20 do not deserve equal confidence."""
    rng = random.Random(11)
    big_pos = [rng.gauss(0.6, 1.0) for _ in range(200)]
    big_neg = [rng.gauss(0.0, 1.0) for _ in range(200)]
    lo_b, hi_b = margin.bootstrap_auc_ci(big_pos, big_neg, seed=1, resamples=400)
    lo_s, hi_s = margin.bootstrap_auc_ci(big_pos[:20], big_neg[:20], seed=1, resamples=400)
    assert (hi_s - lo_s) > (hi_b - lo_b)


def test_no_interval_without_both_arms():
    assert margin.bootstrap_auc_ci([], [1.0], seed=1, resamples=10) is None
    assert margin.bootstrap_auc_ci([1.0], [], seed=1, resamples=10) is None


def test_the_paired_interval_is_tighter_than_two_separate_ones():
    """The point of pairing: shared prompts cancel, so the delta is known far better.

    Two overlapping unpaired intervals do not mean the change is uncertain, and this
    is the test that says so numerically.
    """
    rng = random.Random(3)
    bp = [rng.gauss(0.0, 3.0) for _ in range(150)]          # wide prompt-to-prompt spread
    bn = [rng.gauss(-0.5, 3.0) for _ in range(150)]
    ap = [v + 1.0 for v in bp]                               # the same prompts, shifted
    an = list(bn)
    paired = margin.paired_bootstrap_delta_ci((bp, bn), (ap, an), seed=5, resamples=400)
    paired_width = paired["delta_ci"][1] - paired["delta_ci"][0]

    b_lo, b_hi = margin.bootstrap_auc_ci(bp, bn, seed=5, resamples=400)
    a_lo, a_hi = margin.bootstrap_auc_ci(ap, an, seed=5, resamples=400)
    naive_width = (a_hi - a_lo) + (b_hi - b_lo)              # what eyeballing two intervals implies
    assert paired_width < naive_width


def test_the_paired_delta_matches_the_unresampled_difference():
    bp, bn = [1.0, 2.0, 3.0], [0.0, 0.5, 1.5]
    ap, an = [2.0, 3.0, 4.0], [0.0, 0.5, 1.5]
    got = margin.paired_bootstrap_delta_ci((bp, bn), (ap, an), seed=1, resamples=100)
    assert got["delta_auc"] == pytest.approx(margin.auc(ap, an) - margin.auc(bp, bn), abs=1e-4)


def test_an_unchanged_model_has_a_delta_interval_that_crosses_zero():
    """Identical margins mean no change, and the interval has to be able to say so."""
    rng = random.Random(13)
    pos = [rng.gauss(0.5, 1.0) for _ in range(80)]
    neg = [rng.gauss(0.0, 1.0) for _ in range(80)]
    got = margin.paired_bootstrap_delta_ci((pos, neg), (list(pos), list(neg)), seed=2, resamples=300)
    assert got["delta_auc"] == 0.0
    assert got["delta_crosses_zero"]
    # Sharper than "crosses zero", and the reason this test earns its place: identical
    # inputs can only give an interval of exactly zero width if every replicate applies
    # ONE set of prompt indices to both models. Resample the two arms independently and
    # the deltas scatter, so this is what actually pins the pairing.
    assert got["delta_ci"] == (0.0, 0.0)


@pytest.mark.parametrize(("before", "after"), [
    (([], []), ([1.0], [1.0])),                       # empty
    (([1.0, 2.0], [1.0]), ([1.0], [1.0])),            # harmful arms differ in length
    (([1.0], [1.0, 2.0]), ([1.0], [1.0])),            # harmless arms differ in length
])
def test_the_paired_interval_refuses_mismatched_shapes(before, after):
    assert margin.paired_bootstrap_delta_ci(before, after, seed=1, resamples=10) is None


# ── margins ────────────────────────────────────────────────────────────────────────
def test_margins_returns_one_score_per_prompt(margin_kit):
    model, tok = margin_kit
    hid = margin.label_token_ids(tok, "HARMFUL")
    bid = margin.label_token_ids(tok, "BENIGN")
    prompts = [f"prompt number {i}" for i in range(5)]
    out = margin.margins(model, tok, prompts, hid, bid, "cpu", batch=2)
    assert len(out) == 5
    assert all(isinstance(m, float) for m in out)


def test_margins_are_invariant_to_the_batch_boundary(margin_kit):
    """Batching must not change a score, and the way it would is left-padding.

    The tokenizer left-pads, so the verdict position is the last one for every row
    regardless of how much padding sits in front. Read the wrong index and a short
    prompt in a batch with a long one gets scored on a pad token, which changes
    with batch size. That is exactly what this compares.
    """
    model, tok = margin_kit
    hid = margin.label_token_ids(tok, "HARMFUL")
    bid = margin.label_token_ids(tok, "BENIGN")
    prompts = ["tiny", "a considerably longer prompt than the first one", "mid length here"]
    one = margin.margins(model, tok, prompts, hid, bid, "cpu", batch=1)
    all_at_once = margin.margins(model, tok, prompts, hid, bid, "cpu", batch=16)
    straddling = margin.margins(model, tok, prompts, hid, bid, "cpu", batch=2)
    assert one == pytest.approx(all_at_once, abs=1e-5)
    assert one == pytest.approx(straddling, abs=1e-5)


def test_margins_takes_the_max_over_spellings(margin_kit):
    """Not the first id, and not the mean: the best spelling of each verdict."""
    model, tok = margin_kit
    hid = margin.label_token_ids(tok, "HARMFUL")
    bid = margin.label_token_ids(tok, "BENIGN")
    got = margin.margins(model, tok, ["one prompt"], hid, bid, "cpu")[0]

    texts = [tok.apply_chat_template([{"role": "user", "content": margin.JUDGE_TEMPLATE.format("one prompt")}],
                                     tokenize=False, add_generation_prompt=True)]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    with torch.no_grad():
        logits = model(**enc).logits[:, -1, :].float()
    expected = logits[0, hid].max().item() - logits[0, bid].max().item()
    assert got == pytest.approx(expected, abs=1e-5)


def test_margins_of_no_prompts_is_empty(margin_kit):
    model, tok = margin_kit
    assert margin.margins(model, tok, [], [0], [1], "cpu") == []


# ── the construct-validity controls (task 14) ──────────────────────────────────────
def test_detail_carries_the_margin_the_plain_call_returns(margin_kit):
    """The controls must not be able to disagree with the headline they qualify."""
    model, tok = margin_kit
    hid = margin.label_token_ids(tok, "HARMFUL")
    bid = margin.label_token_ids(tok, "BENIGN")
    prompts = ["short one", "a much longer prompt with a good few more words in it"]
    plain = margin.margins(model, tok, prompts, hid, bid, "cpu", batch=2)
    detailed = margin.margins(model, tok, prompts, hid, bid, "cpu", batch=2, detail=True,
                              canonical=(hid[0], bid[0]))
    assert [r["margin"] for r in detailed] == pytest.approx(plain, abs=1e-6)


def test_the_recorded_length_is_the_real_one_not_the_padded_width(margin_kit):
    """Left padding makes every row the same width, so a padded length is a constant.

    A length control measured on the padded width would report a correlation of nothing
    with everything and look like a clean bill of health.
    """
    model, tok = margin_kit
    hid = margin.label_token_ids(tok, "HARMFUL")
    bid = margin.label_token_ids(tok, "BENIGN")
    prompts = ["tiny", "a considerably longer prompt than the first one by some margin"]
    rows = margin.margins(model, tok, prompts, hid, bid, "cpu", batch=2, detail=True,
                          canonical=(hid[0], bid[0]))
    assert rows[0]["tokens"] < rows[1]["tokens"]

    alone = margin.margins(model, tok, prompts, hid, bid, "cpu", batch=1, detail=True,
                           canonical=(hid[0], bid[0]))
    assert [r["tokens"] for r in rows] == [r["tokens"] for r in alone]


def test_a_verdict_word_with_no_single_token_spelling_says_so(margin_kit):
    model, tok = margin_kit
    hid = margin.label_token_ids(tok, "HARMFUL")
    bid = margin.label_token_ids(tok, "BENIGN")
    rows = margin.margins(model, tok, ["a prompt"], hid, bid, "cpu", detail=True, canonical=None)
    assert rows[0]["canonical"] is None
    out = margin.controls(rows, rows)
    assert out["canonical_auc"] is None and "canonical_note" in out


@pytest.mark.parametrize(("a", "b", "expected"), [
    ([1, 2, 3, 4], [1, 2, 3, 4], 1.0),
    ([1, 2, 3, 4], [4, 3, 2, 1], -1.0),
    # 1 - 6*sum(d^2)/(n(n^2-1)) with d = (0, -2, 1, 1): 1 - 36/60.
    ([1, 2, 3, 4], [1, 4, 2, 3], 0.4),
])
def test_rank_correlation_against_hand_computed_values(a, b, expected):
    assert margin.rank_corr(a, b) == pytest.approx(expected, abs=1e-4)


def test_rank_correlation_averages_ties():
    """Two tied pairs against a monotone series.

    Ranks become 1.5, 1.5, 3.5, 3.5 against 1, 2, 3, 4, and Pearson on those is
    4 / sqrt(4 * 5) = 0.8944. The tie-free version of the same ordering would be 1.0, so
    the number is the tie handling showing up rather than a loss of signal.
    """
    assert margin.rank_corr([1, 1, 2, 2], [1, 2, 3, 4]) == pytest.approx(0.8944, abs=1e-4)


@pytest.mark.parametrize(("a", "b"), [
    ([1, 1, 1], [1, 2, 3]),      # one side constant: no correlation exists to report
    ([1], [1]),                  # too few points
    ([1, 2], [1, 2, 3]),         # mismatched lengths
])
def test_rank_correlation_is_none_where_it_is_undefined(a, b):
    assert margin.rank_corr(a, b) is None


def _rows(margins_, canon, tokens):
    return [{"margin": m, "canonical": c, "tokens": t, "argmax": 0,
             "p_harmful": 0.0, "p_benign": 0.0}
            for m, c, t in zip(margins_, canon, tokens, strict=True)]


# ── the read-out position (task 13f) ───────────────────────────────────────────────
def _readout_rows(argmaxes, p_h=0.4, p_b=0.3):
    return [{"margin": 0.0, "canonical": 0.0, "tokens": 10, "argmax": a,
             "p_harmful": p_h, "p_benign": p_b} for a in argmaxes]


def test_a_verdict_at_the_read_out_position_is_reported_as_such():
    rows = _readout_rows([7, 7, 9, 7])
    out = margin.readout(rows, [7, 9], decode=lambda ids: f"<{ids[0]}>")
    assert out["argmax_is_verdict"] == 1.0
    assert out["top_tokens"][0] == {"id": 7, "text": "<7>", "count": 3}


def test_a_reasoning_opener_at_the_read_out_position_is_caught():
    """The finding this exists for: three of seven published models are thinking models.

    If the most likely token at the scored position is not a verdict, the margin compares two
    tokens the model was never going to emit.
    """
    rows = _readout_rows([151667, 151667, 151667, 7])      # a think-tag three times out of four
    out = margin.readout(rows, [7, 9], decode=lambda ids: "<think>" if ids[0] > 1000 else "H")
    assert out["argmax_is_verdict"] == 0.25
    assert out["top_tokens"][0]["text"] == "<think>"


def test_the_verdict_probability_mass_is_reported():
    """Two logits that together hold a thousandth of the mass are two rounding errors."""
    rows = _readout_rows([1, 1], p_h=0.0004, p_b=0.0002)
    out = margin.readout(rows, [7], decode=lambda ids: "x")
    assert out["verdict_prob_mass_mean"] == pytest.approx(0.0006, abs=1e-9)
    assert out["mean_p_harmful"] == pytest.approx(0.0004, abs=1e-9)
    assert out["mean_p_benign"] == pytest.approx(0.0002, abs=1e-9)


def test_the_read_out_audit_reports_counts_and_never_a_prompt():
    """Aggregate only: a per-prompt list of what the model said is a generation log."""
    out = margin.readout(_readout_rows([1, 2, 3]), [1], decode=lambda ids: "t")
    assert set(out) == {"argmax_is_verdict", "verdict_prob_mass_mean", "verdict_prob_mass_median",
                        "mean_p_harmful", "mean_p_benign", "top_tokens"}
    assert all(set(t) == {"id", "text", "count"} for t in out["top_tokens"])


def test_the_top_token_list_is_capped_and_deterministic():
    rows = _readout_rows(list(range(50)))
    out = margin.readout(rows, [], decode=lambda ids: str(ids[0]))
    assert len(out["top_tokens"]) == margin.READOUT_TOP_TOKENS
    # All tied at one occurrence, so the id breaks the tie and two runs agree.
    assert [t["id"] for t in out["top_tokens"]] == list(range(margin.READOUT_TOP_TOKENS))


def test_no_rows_means_no_read_out_rather_than_a_divide_by_zero():
    assert margin.readout([], [1], decode=lambda ids: "t") is None


def test_the_scored_logits_and_the_read_out_agree(margin_kit):
    """The audit must describe the same forward pass the margin came from, not another one."""
    model, tok = margin_kit
    hid = margin.label_token_ids(tok, "HARMFUL")
    bid = margin.label_token_ids(tok, "BENIGN")
    rows = margin.margins(model, tok, ["one prompt"], hid, bid, "cpu", detail=True,
                          canonical=(hid[0], bid[0]))

    texts = [tok.apply_chat_template(
        [{"role": "user", "content": margin.JUDGE_TEMPLATE.format("one prompt")}],
        tokenize=False, add_generation_prompt=True)]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    with torch.no_grad():
        logits = model(**enc).logits[:, -1, :].float()
    probs = logits.softmax(dim=-1)
    assert rows[0]["argmax"] == int(logits.argmax(dim=-1)[0])
    assert rows[0]["p_harmful"] == pytest.approx(float(probs[0, hid].sum()), abs=1e-6)
    assert rows[0]["p_benign"] == pytest.approx(float(probs[0, bid].sum()), abs=1e-6)


def test_a_length_only_ruler_is_scored_on_the_same_scale_as_the_headline():
    """If the harmful arm is simply longer, a ruler that reads nothing scores well.

    This is the control's whole point: a length-only AUC near the headline means the
    headline may be reporting a property of the corpus rather than of the model.
    """
    harmful = _rows([5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [40, 45, 50])
    harmless = _rows([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [10, 12, 14])
    out = margin.controls(harmful, harmless)
    assert out["length_only_auc"] == 1.0            # perfectly separable by length alone
    assert out["mean_tokens_harmful"] == 45.0
    assert out["mean_tokens_harmless"] == 12.0


def test_a_length_balanced_corpus_gives_a_length_only_auc_at_chance():
    harmful = _rows([5.0, 6.0], [5.0, 6.0], [30, 10])
    harmless = _rows([1.0, 2.0], [1.0, 2.0], [10, 30])
    assert margin.controls(harmful, harmless)["length_only_auc"] == 0.5


def test_the_canonical_auc_can_disagree_with_the_headline_auc():
    """Which is the finding: taking the max over spellings is a choice, not a measurement."""
    harmful = _rows([9.0, 9.0], [-1.0, -2.0], [20, 20])
    harmless = _rows([1.0, 1.0], [5.0, 6.0], [20, 20])
    out = margin.controls(harmful, harmless)
    assert out["canonical_auc"] == 0.0              # the fixed token pair inverts it entirely


def test_the_controls_report_no_prompts(margin_kit):
    """Every diagnostic here is a number, because the inputs are harmful text."""
    harmful = _rows([1.0, 2.0], [1.0, 2.0], [10, 20])
    out = margin.controls(harmful, harmful)
    assert all(v is None or isinstance(v, (int, float, str)) for v in out.values())
    assert not any(isinstance(v, str) and " " in v and "tokenizer" not in v
                   for k, v in out.items() if k != "canonical_note")


# ── the memory fix (task 10) ───────────────────────────────────────────────────────
def test_only_the_last_position_is_computed(margin_kit, monkeypatch):
    """A [B, T, V] tensor is ~10 GB at batch 16 on a large vocabulary, all but one row wasted."""
    model, tok = margin_kit
    seen = {}
    real = model.forward

    def spy(**kw):
        seen["logits_to_keep"] = kw.get("logits_to_keep")
        out = real(**kw)
        seen["shape"] = tuple(out.logits.shape)
        return out

    monkeypatch.setattr(model, "forward", spy)
    margin.margins(model, tok, ["a prompt", "another prompt"], [1], [2], "cpu", batch=2)
    assert seen["logits_to_keep"] == 1
    assert seen["shape"][1] == 1, "the model returned more than one position of logits"


def test_the_score_is_unchanged_by_the_memory_fix(margin_kit):
    """The shape contract holds: [B, 1, V] and [B, T, V] give the same [:, -1, :] row."""
    model, tok = margin_kit
    hid, bid = margin.label_token_ids(tok, "HARMFUL"), margin.label_token_ids(tok, "BENIGN")
    prompts = ["tiny", "a considerably longer prompt than the first one"]
    with_fix = margin.margins(model, tok, prompts, hid, bid, "cpu", batch=2)

    class _FullLogitsOnly(type(model)):
        def forward(self, **kw):
            if kw.pop("logits_to_keep", None) is not None:
                pass                       # deliberately ignore it: compute everything
            return super().forward(**kw)

    model.__class__ = _FullLogitsOnly
    assert margin.margins(model, tok, prompts, hid, bid, "cpu", batch=2) == pytest.approx(with_fix, abs=1e-6)


def test_a_model_that_rejects_the_argument_falls_back_loudly(margin_kit, capsys):
    """Visible degradation, not silent: the fallback needs far more memory."""
    from senbonzakura import cli
    model, tok = margin_kit

    class _Rejects(type(model)):
        def forward(self, **kw):
            if "logits_to_keep" in kw:
                raise TypeError("forward() got an unexpected keyword argument 'logits_to_keep'")
            return super().forward(**kw)

    model.__class__ = _Rejects
    cli._NO_LOGITS_TO_KEEP.discard(_Rejects.__name__)
    out = margin.margins(model, tok, ["a", "b", "c"], [1], [2], "cpu", batch=1)
    assert len(out) == 3
    assert "does not accept logits_to_keep" in capsys.readouterr().out
    # Warned once for the class, not once per batch.
    assert _Rejects.__name__ in cli._NO_LOGITS_TO_KEEP
    cli._NO_LOGITS_TO_KEEP.discard(_Rejects.__name__)


def test_an_unrelated_type_error_is_not_swallowed(margin_kit):
    """Converting a real forward-pass bug into a quiet memory change would hide it."""
    model, tok = margin_kit

    class _Broken(type(model)):
        def forward(self, **kw):
            raise TypeError("something else entirely is wrong")

    model.__class__ = _Broken
    with pytest.raises(TypeError, match="something else entirely"):
        margin.margins(model, tok, ["a"], [1], [2], "cpu")


# ── main ───────────────────────────────────────────────────────────────────────────
def _track(tmp_path, n_harmful=4, n_harmless=6):
    from datasets import Dataset
    bad = str(tmp_path / "harmful")
    good = str(tmp_path / "harmless")
    Dataset.from_dict({"text": [f"harmful request {i}" for i in range(n_harmful)]}).save_to_disk(bad)
    Dataset.from_dict({"text": [f"harmless question {i}" for i in range(n_harmless)]}).save_to_disk(good)
    return bad, good


@pytest.fixture
def loaded(monkeypatch, margin_kit):
    model, tok = margin_kit
    monkeypatch.setattr(margin, "load_model_and_tokenizer", lambda *a, **k: (model, tok))
    return model, tok


def test_main_end_to_end(loaded, tmp_path):
    bad, good = _track(tmp_path)
    out = str(tmp_path / "res.json")
    res = margin.main(["--model", "x", "--harmful", bad, "--harmless", good, "--out", out,
                       "--n", "3", "--skip-harmful", "0", "--skip-harmless", "0",
                       "--device", "cpu", "--label", "after"])
    assert res["mode"] == "logit_margin"
    assert res["label"] == "after"
    assert res["n_harmful"] == 3
    assert res["n_harmless"] == 3
    assert 0.0 <= res["auc"] <= 1.0
    assert 0.0 <= res["frac_harmful_positive"] <= 1.0
    with open(out, encoding="utf-8") as f:
        assert json.load(f)["auc"] == res["auc"]


def test_main_writes_every_per_prompt_margin(loaded, tmp_path):
    """The percentages hid both historical scoring bugs; the per-prompt rows did not."""
    bad, good = _track(tmp_path)
    mpath = str(tmp_path / "m.jsonl")
    margin.main(["--model", "x", "--harmful", bad, "--harmless", good, "--out", str(tmp_path / "r.json"),
                 "--margins", mpath, "--n", "2", "--skip-harmful", "0", "--skip-harmless", "0", "--device", "cpu"])
    with open(mpath, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 4
    assert [r["set"] for r in rows] == ["harmful", "harmful", "harmless", "harmless"]
    assert [r["i"] for r in rows] == [0, 1, 0, 1]
    assert rows[0]["prompt"] == "harmful request 0"
    assert all(isinstance(r["margin"], float) for r in rows)


def test_main_holds_out_the_head_of_the_harmless_set(loaded, tmp_path):
    """--skip-harmless exists because the directions were fitted on that head."""
    bad, good = _track(tmp_path, n_harmless=8)
    mpath = str(tmp_path / "m.jsonl")
    margin.main(["--model", "x", "--harmful", bad, "--harmless", good, "--out", str(tmp_path / "r.json"),
                 "--margins", mpath, "--n", "2", "--skip-harmful", "0", "--skip-harmless", "5", "--device", "cpu"])
    with open(mpath, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    harmless = [r["prompt"] for r in rows if r["set"] == "harmless"]
    assert harmless == ["harmless question 5", "harmless question 6"]


def test_margins_are_retained_by_default(loaded, tmp_path):
    """Both historical scoring bugs were invisible in the percentages, so the default is on."""
    bad, good = _track(tmp_path)
    out = str(tmp_path / "deep" / "res.json")
    Path(out).parent.mkdir()
    res = margin.main(["--model", "x", "--harmful", bad, "--harmless", good, "--out", out,
                       "--n", "2", "--skip-harmful", "0", "--skip-harmless", "0", "--device", "cpu"])
    expected = tmp_path / "deep" / "res.margins.jsonl"
    assert res["margins_path"] == str(expected)
    assert expected.exists()
    assert len(expected.read_text(encoding="utf-8").strip().split("\n")) == 4


def test_no_margins_turns_retention_off(loaded, tmp_path):
    """An explicit empty string must not be overwritten by the default."""
    bad, good = _track(tmp_path)
    out = str(tmp_path / "res.json")
    res = margin.main(["--model", "x", "--harmful", bad, "--harmless", good, "--out", out,
                       "--no-margins", "--n", "2", "--skip-harmful", "0", "--skip-harmless", "0",
                       "--device", "cpu"])
    assert res["margins_path"] == ""
    assert not (tmp_path / "res.margins.jsonl").exists()


def test_the_retained_rows_would_be_refused_by_the_commit_guard(loaded, tmp_path):
    """The two halves of task 11 meet here: retention is on, and the guard catches it.

    A default-on artefact full of harmful prompts is only safe because something
    refuses to commit it. This asserts the guard actually fires on what margin.py
    writes, rather than on a hand-built example of what it might write.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cpa", Path(__file__).resolve().parent.parent / "tools" / "check_prompt_artefacts.py")
    cpa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpa)

    bad, good = _track(tmp_path)
    margin.main(["--model", "x", "--harmful", bad, "--harmless", good,
                 "--out", str(tmp_path / "res.json"), "--n", "2", "--skip-harmful", "0",
                 "--skip-harmless", "0", "--device", "cpu"])
    findings = cpa.scan_file(tmp_path / "res.margins.jsonl")
    assert findings, "the guard did not fire on a retained-margins file"
    assert all("carries prompt" in f for f in findings)


def test_main_holds_out_the_head_of_the_harmful_set_too(loaded, tmp_path):
    """The search selected its winning trial on the head of the harmful set.

    Scoring the compass there measures the prompts the surgery was tuned against,
    which is the same mistake --skip-harmless already existed to avoid on the other
    arm.
    """
    bad, good = _track(tmp_path, n_harmful=8)
    mpath = str(tmp_path / "m.jsonl")
    margin.main(["--model", "x", "--harmful", bad, "--harmless", good, "--out", str(tmp_path / "r.json"),
                 "--margins", mpath, "--n", "2", "--skip-harmful", "5", "--skip-harmless", "0",
                 "--device", "cpu"])
    with open(mpath, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert [r["prompt"] for r in rows if r["set"] == "harmful"] == ["harmful request 5", "harmful request 6"]


def test_main_records_both_skips(loaded, tmp_path):
    """"Held out" is a claim about these two numbers, and the AUC does not carry them."""
    bad, good = _track(tmp_path, n_harmful=8, n_harmless=8)
    res = margin.main(["--model", "x", "--harmful", bad, "--harmless", good,
                       "--out", str(tmp_path / "r.json"), "--n", "2",
                       "--skip-harmful", "3", "--skip-harmless", "4", "--device", "cpu"])
    assert res["skip_harmful"] == 3
    assert res["skip_harmless"] == 4


def test_the_default_skips_hold_out_both_arms():
    """A default of 0 on either arm would publish a selection-set number by accident."""
    a = margin.build_parser().parse_args(["--model", "m", "--harmful", "h", "--harmless", "l", "--out", "o"])
    assert a.skip_harmless == 320     # dir_prompts 256 plus the drift check
    assert a.skip_harmful == 128      # the largest --eval-refusal-final any auto preset uses


@pytest.mark.parametrize(("flag", "n_harmful", "n_harmless"), [
    ("--skip-harmless", 8, 6),
    ("--skip-harmful", 6, 8),
])
def test_main_refuses_a_skip_that_does_not_fit_on_either_arm(loaded, tmp_path, flag, n_harmful, n_harmless):
    """The harmful arm used to score short and say nothing about it."""
    bad, good = _track(tmp_path, n_harmful=n_harmful, n_harmless=n_harmless)
    other = "--skip-harmful" if flag == "--skip-harmless" else "--skip-harmless"
    with pytest.raises(SystemExit, match="does not fit"):
        margin.main(["--model", "x", "--harmful", bad, "--harmless", good,
                     "--out", str(tmp_path / "r.json"), "--n", "4", flag, "4", other, "0",
                     "--device", "cpu"])


def test_main_refuses_a_tokenizer_that_shares_a_verdict_id(loaded, tmp_path, monkeypatch):
    """Shared ids attenuate every margin, which reads as a model that cannot tell."""
    bad, good = _track(tmp_path)
    monkeypatch.setattr(margin, "label_token_ids", lambda tok, word: [3, 7])
    with pytest.raises(SystemExit, match="shared first-token ids"):
        margin.main(["--model", "x", "--harmful", bad, "--harmless", good,
                     "--out", str(tmp_path / "r.json"), "--n", "2", "--skip-harmful", "0",
                     "--skip-harmless", "0", "--device", "cpu"])


# ── the controls, end to end (task 14) ─────────────────────────────────────────────
def test_every_result_carries_the_read_out_audit(loaded, tmp_path, capsys):
    res = _run(loaded, tmp_path, "readout")
    for arm in ("harmful", "harmless"):
        r = res["readout"][arm]
        assert 0.0 <= r["argmax_is_verdict"] <= 1.0
        assert r["top_tokens"]
    assert "MARGIN_READOUT" in capsys.readouterr().out


def test_every_result_carries_the_controls(loaded, tmp_path, capsys):
    """No refusal or discrimination figure ships without the controls that qualify it."""
    res = _run(loaded, tmp_path, "controlled")
    c = res["controls"]
    for key in ("length_only_auc", "canonical_auc", "length_corr_harmful",
                "length_corr_harmless", "mean_tokens_harmful", "mean_tokens_harmless"):
        assert key in c, f"the controls lost {key}"
    assert "MARGIN_CONTROLS" in capsys.readouterr().out


def test_the_topic_matched_arm_is_reported_beside_the_headline(loaded, tmp_path, capsys):
    from datasets import Dataset
    matched = str(tmp_path / "matched")
    Dataset.from_dict({"text": [f"harmless question about the same subject {i}" for i in range(4)]}
                      ).save_to_disk(matched)
    res = _run(loaded, tmp_path, "matched", extra=["--harmless-matched", matched])
    t = res["topic_matched"]
    assert t["n"] == 3 and t["source"] == matched
    assert t["auc_ci"][0] <= t["auc"] <= t["auc_ci"][1]
    assert "controls" in t
    out = capsys.readouterr().out
    assert "MARGIN_TOPIC_MATCHED" in out
    assert "on the unmatched harmless arm" in out       # the comparison, not a lone number


def test_a_topic_matched_skip_that_leaves_nothing_is_refused(loaded, tmp_path):
    from datasets import Dataset
    matched = str(tmp_path / "matched")
    Dataset.from_dict({"text": ["one matched harmless prompt"]}).save_to_disk(matched)
    with pytest.raises(SystemExit, match="leaves none"):
        _run(loaded, tmp_path, "matched",
             extra=["--harmless-matched", matched, "--skip-matched", "5"])


def test_a_run_without_a_matched_set_says_nothing_about_one(loaded, tmp_path, capsys):
    res = _run(loaded, tmp_path, "plain")
    assert "topic_matched" not in res
    assert "MARGIN_TOPIC_MATCHED" not in capsys.readouterr().out


# ── the paired comparison end to end ───────────────────────────────────────────────
def _run(loaded, tmp_path, tag, extra=()):
    bad, good = _track(tmp_path, n_harmful=4, n_harmless=4)
    out = str(tmp_path / f"{tag}.json")
    return margin.main(["--model", "x", "--harmful", bad, "--harmless", good, "--out", out,
                        "--n", "3", "--skip-harmful", "0", "--skip-harmless", "0",
                        "--bootstrap", "60", "--device", "cpu", "--label", tag, *extra])


def test_the_result_carries_an_interval_and_its_seed(loaded, tmp_path, capsys):
    res = _run(loaded, tmp_path, "before")
    lo, hi = res["auc_ci"]
    assert lo <= res["auc"] <= hi
    assert res["bootstrap_resamples"] == 60
    assert res["seed"] == 42
    assert "ci=[" in capsys.readouterr().out


def test_bootstrap_zero_skips_the_interval(loaded, tmp_path):
    res = _run(loaded, tmp_path, "nb", extra=["--bootstrap", "0"])
    assert res["auc_ci"] is None


def test_compare_to_adds_the_paired_interval(loaded, tmp_path, capsys):
    """The before-and-after shape the writeup's table needs."""
    before = _run(loaded, tmp_path, "before")
    after = _run(loaded, tmp_path, "after", extra=["--compare-to", before["margins_path"]])
    paired = after["paired"]
    assert paired["delta_auc"] == 0.0            # the same fixture model both times
    assert paired["delta_crosses_zero"]
    assert after["compared_to"] == before["margins_path"]
    assert "MARGIN_PAIRED" in capsys.readouterr().out


def test_compare_to_refuses_a_file_scoring_different_prompts(loaded, tmp_path):
    """A tight interval around a meaningless difference is worse than no interval."""
    before = _run(loaded, tmp_path, "before")
    rows = [json.loads(x) for x in Path(before["margins_path"]).read_text(encoding="utf-8").splitlines()]
    rows[1]["prompt"] = "a completely different prompt"
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(SystemExit, match="not paired"):
        _run(loaded, tmp_path, "after", extra=["--compare-to", str(tampered)])


def test_compare_to_refuses_a_file_with_the_wrong_row_count(loaded, tmp_path):
    before = _run(loaded, tmp_path, "before")
    rows = Path(before["margins_path"]).read_text(encoding="utf-8").splitlines()
    short = tmp_path / "short.jsonl"
    short.write_text("\n".join(rows[:-1]), encoding="utf-8")
    with pytest.raises(SystemExit, match="paired interval needs the same prompts"):
        _run(loaded, tmp_path, "after", extra=["--compare-to", str(short)])


def test_compare_to_refuses_a_missing_file(loaded, tmp_path):
    with pytest.raises(SystemExit, match="could not read --compare-to"):
        _run(loaded, tmp_path, "after", extra=["--compare-to", str(tmp_path / "absent.jsonl")])


def test_compare_to_refuses_malformed_rows(loaded, tmp_path):
    bad_rows = tmp_path / "bad.jsonl"
    bad_rows.write_text('{"i": 0, "set": "harmful", "margin": 1.0}\nnot json at all\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="not valid JSON"):
        _run(loaded, tmp_path, "after", extra=["--compare-to", str(bad_rows)])


def test_compare_to_refuses_an_unknown_arm_label(loaded, tmp_path):
    odd = tmp_path / "odd.jsonl"
    odd.write_text('{"i": 0, "set": "neither", "margin": 1.0}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="expected harmful or harmless"):
        _run(loaded, tmp_path, "after", extra=["--compare-to", str(odd)])


def test_compare_to_refuses_a_gap_in_the_row_indices(loaded, tmp_path):
    """Right count, wrong indices: the rows do not cover the prompts they claim to."""
    before = _run(loaded, tmp_path, "before")
    rows = [json.loads(x) for x in Path(before["margins_path"]).read_text(encoding="utf-8").splitlines()]
    for r in rows:
        if r["set"] == "harmful" and r["i"] == 1:
            r["i"] = 99
    gapped = tmp_path / "gapped.jsonl"
    gapped.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(SystemExit, match="missing harmful row 1"):
        _run(loaded, tmp_path, "after", extra=["--compare-to", str(gapped)])


def test_compare_to_refuses_a_row_whose_margin_is_not_a_number(loaded, tmp_path):
    """Otherwise it reaches the bootstrap as None and dies there, naming no row."""
    before = _run(loaded, tmp_path, "before")
    rows = [json.loads(x) for x in Path(before["margins_path"]).read_text(encoding="utf-8").splitlines()]
    del rows[2]["margin"]
    holed = tmp_path / "holed.jsonl"
    holed.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(SystemExit, match="is not a number"):
        _run(loaded, tmp_path, "after", extra=["--compare-to", str(holed)])


def test_compare_to_with_no_resampling_is_refused_rather_than_overridden(loaded, tmp_path):
    """--bootstrap 0 asked for no resampling; the paired interval is resampling."""
    before = _run(loaded, tmp_path, "before")
    with pytest.raises(SystemExit, match="pick one"):
        _run(loaded, tmp_path, "after",
             extra=["--compare-to", before["margins_path"], "--bootstrap", "0"])


# ── arguments that would silently score the wrong rows ─────────────────────────────
@pytest.mark.parametrize(("extra", "expected"), [
    (["--n", "0"], "scores no prompts"),
    (["--n", "-3"], "scores no prompts"),
    (["--skip-harmful", "-1"], "is negative"),
    (["--skip-harmless", "-1"], "is negative"),
])
def test_nonsense_slice_arguments_are_refused(loaded, tmp_path, extra, expected):
    """A negative skip reads the TAIL of the set: real rows, from the wrong partition."""
    bad, good = _track(tmp_path, n_harmful=4, n_harmless=4)
    with pytest.raises(SystemExit, match=expected):
        margin.main(["--model", "x", "--harmful", bad, "--harmless", good,
                     "--out", str(tmp_path / "never.json"), "--n", "3",
                     "--skip-harmful", "0", "--skip-harmless", "0",
                     "--bootstrap", "0", "--device", "cpu", *extra])


# ── the dataset boundary ───────────────────────────────────────────────────────────
def test_an_empty_dataset_names_itself(loaded, tmp_path):
    """load_from_disk raises a bare IndexError on an empty set, naming neither file nor flag.

    The wrapper does not add its own empty check: the raise happens inside
    load_from_disk before any row is read, so what this asserts is that the
    IndexError comes back out saying which dataset and which flag it was.
    """
    from datasets import Dataset
    _, good = _track(tmp_path)
    empty = str(tmp_path / "empty")
    Dataset.from_dict({"text": []}).save_to_disk(empty)
    # The message differs across the supported datasets range (5.x raises inside
    # load_from_disk, 2.15 returns nothing), so assert what both paths guarantee:
    # a loud exit that names which arm.
    with pytest.raises(SystemExit, match="harmful"):
        margin.main(["--model", "x", "--harmful", empty, "--harmless", good,
                     "--out", str(tmp_path / "r.json"), "--n", "1", "--skip-harmful", "0",
                     "--skip-harmless", "0", "--device", "cpu"])


def test_a_dataset_without_a_text_column_says_which_columns_it_has(loaded, tmp_path):
    from datasets import Dataset
    bad, _ = _track(tmp_path)
    wrong = str(tmp_path / "wrong")
    Dataset.from_dict({"prompt": ["a", "b"]}).save_to_disk(wrong)
    with pytest.raises(SystemExit, match="no 'text' column"):
        margin.main(["--model", "x", "--harmful", bad, "--harmless", wrong,
                     "--out", str(tmp_path / "r.json"), "--n", "1", "--skip-harmful", "0",
                     "--skip-harmless", "0", "--device", "cpu"])


def test_a_missing_dataset_directory_names_the_path(loaded, tmp_path):
    _, good = _track(tmp_path)
    with pytest.raises(SystemExit, match="could not load the harmful dataset"):
        margin.main(["--model", "x", "--harmful", str(tmp_path / "nope"), "--harmless", good,
                     "--out", str(tmp_path / "r.json"), "--n", "1", "--skip-harmful", "0",
                     "--skip-harmless", "0", "--device", "cpu"])


def test_main_refuses_a_tokenizer_whose_verdict_words_do_not_resolve(loaded, tmp_path, monkeypatch):
    bad, good = _track(tmp_path)
    monkeypatch.setattr(margin, "label_token_ids", lambda tok, word: [])
    with pytest.raises(SystemExit, match="verdict token ids"):
        margin.main(["--model", "x", "--harmful", bad, "--harmless", good,
                     "--out", str(tmp_path / "r.json"), "--n", "2", "--skip-harmful", "0", "--skip-harmless", "0",
                     "--device", "cpu"])


def test_a_dataset_that_loads_but_holds_nothing_is_refused(loaded, tmp_path, monkeypatch):
    """Version-independent cover for the empty branch.

    datasets 5.x raises inside load_from_disk before a row is read, so on a current
    install that branch is unreachable; on the 2.15 floor the same directory loads and
    yields nothing. Faking the loader exercises it on every supported version.
    """
    class _Empty:
        column_names = ("text",)

        def __iter__(self):
            return iter(())

    bad, good = _track(tmp_path)
    monkeypatch.setattr(margin, "load_from_disk", lambda _p: _Empty())
    with pytest.raises(SystemExit, match=r"harmful dataset .* is empty"):
        margin.main(["--model", "x", "--harmful", bad, "--harmless", good,
                     "--out", str(tmp_path / "r.json"), "--n", "1", "--skip-harmful", "0",
                     "--skip-harmless", "0", "--device", "cpu"])


# ── provenance (task 13) ───────────────────────────────────────────────────────────
def test_the_result_records_which_tokens_were_scored(loaded, tmp_path, monkeypatch):
    """The AUC is a difference between two logits, and nothing else says which two.

    A tokenizer change that moves HARMFUL onto a different first token produces a
    perfectly plausible number for a different measurement, and without this there is
    no way to tell that happened.
    """
    _model, tok = loaded
    monkeypatch.setattr(tok, "decode", lambda ids, **_k: f"<tok{ids[0]}>")
    bad, good = _track(tmp_path)
    res = margin.main(["--model", "x", "--harmful", bad, "--harmless", good,
                       "--out", str(tmp_path / "r.json"), "--n", "2", "--skip-harmful", "0",
                       "--skip-harmless", "0", "--bootstrap", "0", "--device", "cpu"])
    tokens = res["verdict_tokens"]
    assert set(tokens) == {"HARMFUL", "BENIGN"}
    for side in tokens.values():
        assert side, "an empty verdict token set would mean nothing was scored"
        for entry in side:
            assert isinstance(entry["id"], int)
            # Decoded from THAT id. The fixture tokenizer's decode returns a constant, so
            # asserting the value is merely a string would pass even if the text were
            # unrelated to the token, which is the only thing this field is for.
            assert entry["text"] == f"<tok{entry['id']}>"
    ids = {e["id"] for e in tokens["HARMFUL"]} & {e["id"] for e in tokens["BENIGN"]}
    assert not ids, "the two sides must not share a scored token"


def test_the_result_records_the_environment_that_produced_it(loaded, tmp_path):
    bad, good = _track(tmp_path)
    res = margin.main(["--model", "x", "--harmful", bad, "--harmless", good,
                       "--out", str(tmp_path / "r.json"), "--n", "2", "--skip-harmful", "0",
                       "--skip-harmless", "0", "--bootstrap", "0", "--device", "cpu"])
    p = res["provenance"]
    assert p["device"] == "cpu"
    assert p["packages"]["torch"]
    assert p["senbonzakura"]["version"]


def test_everything_recorded_survives_the_round_trip_to_disk(loaded, tmp_path):
    """A field that cannot be serialised is a lost run, discovered at the very end."""
    bad, good = _track(tmp_path)
    out = tmp_path / "r.json"
    res = margin.main(["--model", "x", "--harmful", bad, "--harmless", good, "--out", str(out),
                       "--n", "2", "--skip-harmful", "0", "--skip-harmless", "0",
                       "--bootstrap", "20", "--device", "cpu"])
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["verdict_tokens"] == res["verdict_tokens"]
    assert on_disk["provenance"]["packages"] == res["provenance"]["packages"]


def test_the_recorded_fields_answer_the_questions_a_rerun_asks(loaded, tmp_path):
    """The July sweep is unreproducible because these were never written down."""
    bad, good = _track(tmp_path)
    res = margin.main(["--model", "x", "--harmful", bad, "--harmless", good,
                       "--out", str(tmp_path / "r.json"), "--n", "2", "--skip-harmful", "0",
                       "--skip-harmless", "0", "--bootstrap", "0", "--device", "cpu"])
    required = {"model", "seed", "skip_harmful", "skip_harmless", "n_harmful", "n_harmless",
                "chat_template", "verdict_tokens", "provenance", "bootstrap_resamples"}
    assert required <= set(res), f"missing: {sorted(required - set(res))}"
