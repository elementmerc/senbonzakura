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
    with pytest.raises(SystemExit, match="could not load the harmful dataset"):
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
