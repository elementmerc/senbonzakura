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
                       "--n", "3", "--skip-harmless", "0", "--device", "cpu", "--label", "after"])
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
                 "--margins", mpath, "--n", "2", "--skip-harmless", "0", "--device", "cpu"])
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
                 "--margins", mpath, "--n", "2", "--skip-harmless", "5", "--device", "cpu"])
    with open(mpath, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    harmless = [r["prompt"] for r in rows if r["set"] == "harmless"]
    assert harmless == ["harmless question 5", "harmless question 6"]


def test_main_refuses_a_harmless_skip_that_does_not_fit(loaded, tmp_path):
    bad, good = _track(tmp_path, n_harmless=6)
    with pytest.raises(SystemExit, match="does not fit"):
        margin.main(["--model", "x", "--harmful", bad, "--harmless", good,
                     "--out", str(tmp_path / "r.json"), "--n", "4", "--skip-harmless", "4",
                     "--device", "cpu"])


def test_main_refuses_a_tokenizer_whose_verdict_words_do_not_resolve(loaded, tmp_path, monkeypatch):
    bad, good = _track(tmp_path)
    monkeypatch.setattr(margin, "label_token_ids", lambda tok, word: [])
    with pytest.raises(SystemExit, match="verdict token ids"):
        margin.main(["--model", "x", "--harmful", bad, "--harmless", good,
                     "--out", str(tmp_path / "r.json"), "--n", "2", "--skip-harmless", "0",
                     "--device", "cpu"])
