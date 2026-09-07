# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The receipt this project did not have: what the edit cost in capability.

WHY THIS FILE EXISTS

Verified in the source on 2026-09-07: nothing in `src/` or `tools/` measured capability at all.
Every ruler in the pipeline is about refusal, `broken_rate` catches a model that has stopped
forming sentences rather than one that forms them and can no longer add up, and KL is a
distributional proxy taken on harmless prompts. A real run reported KL 0.128 with `broken=0%`, and
both of those are compatible with losing several points of multi-step arithmetic, because nothing
ever asked the model to do any.

THE TRAP EVERY TEST HERE IS ABOUT

An answer that could not be graded must be INDETERMINATE, never wrong. If truncation scored as
incorrect, then a small `--max-new` would look exactly like capability loss, and the error would
point in the same direction as the claim the module exists to test, which is the worst direction
available. The same defect in a different costume reached a published figure here once, reading a
thinking model's verdict at the position it emits `<think>`.
"""
import pytest

from senbonzakura import capability as cap

# ── reading a number the way a model writes one ──────────────────────────────────────

@pytest.mark.parametrize(("text", "want"), [
    ("42", 42.0),
    ("1,234", 1234.0),           # models emit thousands separators, the gold answers do not
    ("12.0", 12.0),
    ("-5", -5.0),
    ("18.", 18.0),               # sentence-ending full stop
    ("", None),
    ("twelve", None),
    (None, None),
])
def test_numbers_are_read_as_written(text, want):
    assert cap.parse_number(text) == want


def test_the_gold_answer_comes_from_after_the_marker():
    """GSM8K solutions are worked text with the answer after ####, and the working is full of
    other numbers. Taking the last number of the whole solution would usually be right and
    sometimes silently wrong.
    """
    solution = "He had 5, bought 3 more, then gave 2 away.\n#### 6"
    assert cap.gold_answer(solution) == 6.0


def test_a_solution_with_no_marker_still_yields_its_last_number():
    assert cap.gold_answer("the answer is 6") == 6.0


def test_a_row_with_no_number_at_all_is_unusable():
    assert cap.gold_answer("no digits here") is None


def test_the_prediction_is_the_last_number_not_the_first():
    """A worked solution restates the question's quantities before reaching a result."""
    assert cap.predicted_answer("He had 5 apples and bought 3, so he has 8") == 8.0


# ── grading, and what must never be scored wrong ─────────────────────────────────────

@pytest.mark.parametrize(("gen", "want"), [
    ("The answer is 8", "correct"),
    ("The answer is 9", "wrong"),
    ("", "indeterminate"),
    ("   ", "indeterminate"),
    ("I cannot help with that.", "indeterminate"),
    ("The answer is eight", "indeterminate"),
])
def test_grading_one_item(gen, want):
    assert cap.grade_one(gen, "#### 8") == want


def test_a_refusal_is_not_a_wrong_answer():
    """An abliterated model refusing a maths question has not lost arithmetic, and counting it
    wrong would let a refusal rate leak into a capability number.
    """
    assert cap.grade_one("I'm sorry, I can't help with that.", "#### 8") == "indeterminate"


def test_a_truncated_answer_is_indeterminate_even_when_it_contains_numbers():
    """THE REGRESSION THIS FILE IS NAMED FOR, and the one text alone cannot catch.

    Cut off mid-solution, the last number the model managed to write is not its answer. Without
    the flag this scores WRONG, and every such item is a token budget reported as capability loss.
    """
    cut = "He had 5 apples and bought 3 more, so he"
    assert cap.grade_one(cut, "#### 8") == "wrong", (
        "if text alone now detects this, the flag may be redundant; check before removing it")
    assert cap.grade_one(cut, "#### 8", truncated=True) == "indeterminate"


def test_a_correct_answer_that_was_truncated_is_still_indeterminate():
    """Not "correct if it happens to match". A truncated generation whose last number is right by
    luck is not evidence the model can do the sum.
    """
    assert cap.grade_one("5 plus 3 is 8 and then", "#### 8", truncated=True) == "indeterminate"


def test_an_unusable_dataset_row_is_not_the_models_fault():
    assert cap.grade_one("The answer is 8", "no gold here") == "indeterminate"


def test_grading_a_batch_requires_matching_lengths():
    with pytest.raises(ValueError):
        cap.grade(["a", "b"], ["#### 1"])


# ── the summary keeps the two rates apart ────────────────────────────────────────────

def test_accuracy_is_over_what_could_be_graded():
    """Folding indeterminates into the denominator would let a small token budget depress the
    accuracy of a model that is perfectly capable.

    Sized above the reporting floor on purpose. The first version used four items and started
    failing when the floor landed, which is the floor working: an accuracy over three graded items
    is not an accuracy. What is under test here is the DENOMINATOR, so the sample is large enough
    for a rate to exist and the indeterminates still must not count towards it.
    """
    s = cap.summarise(["correct"] * 40 + ["wrong"] * 20 + ["indeterminate"] * 15)
    assert s["graded"] == 60, "indeterminates must not enter the denominator"
    assert s["accuracy"] == pytest.approx(40 / 60, abs=1e-4)
    assert s["indeterminate"] == 15
    assert s["n"] == 75


def test_the_counts_are_reported_not_only_the_rate():
    """A rate over a handful of observations is what reversed direction on the ROG when the sample
    grew. Counts cannot be quoted as something they are not.
    """
    s = cap.summarise(["correct", "wrong"])
    assert s["correct"] == 1 and s["wrong"] == 1 and s["n"] == 2


def test_a_run_that_graded_nothing_reports_no_accuracy_rather_than_zero():
    """Zero would read as "got everything wrong", which is the most alarming possible
    substitution for "measured nothing".
    """
    s = cap.summarise(["indeterminate", "indeterminate"])
    assert s["accuracy"] is None
    assert s["indeterminate_rate"] == 1.0


# ── the paired comparison ────────────────────────────────────────────────────────────

def _verdicts(pattern):
    return [{"c": "correct", "w": "wrong", "i": "indeterminate"}[ch] for ch in pattern]


def test_a_real_drop_is_measured_and_its_sign_is_right():
    before = _verdicts("c" * 80 + "w" * 20)
    after = _verdicts("c" * 60 + "w" * 40)
    change = cap.paired_change(before, after, resamples=400)
    assert change["delta_accuracy"] == pytest.approx(-0.20, abs=1e-6)
    assert change["items_broken"] == 20
    assert change["items_fixed"] == 0
    assert change["distinguishable_from_zero"] is True


def test_no_change_is_reported_as_not_distinguishable_from_zero():
    """The sentence the field's published figures are missing."""
    v = _verdicts("cw" * 50)
    change = cap.paired_change(v, list(v), resamples=400)
    assert change["delta_accuracy"] == 0.0
    assert change["distinguishable_from_zero"] is False


def test_items_moving_both_ways_are_reported_separately():
    """A delta cannot say whether a model traded some items for others, and that is a different
    thing from getting uniformly worse.
    """
    before = _verdicts("ccww")
    after = _verdicts("wwcc")
    change = cap.paired_change(before, after, resamples=200)
    assert change["items_broken"] == 2
    assert change["items_fixed"] == 2
    assert change["delta_accuracy"] == 0.0


def test_items_indeterminate_on_either_side_are_dropped_not_scored():
    """A pair where one side could not be graded says nothing about the difference between them."""
    before = _verdicts("cci")
    after = _verdicts("ccc")
    change = cap.paired_change(before, after, resamples=200)
    assert change["compared_on"] == 2
    assert change["dropped_indeterminate"] == 1


def test_mismatched_lengths_refuse_rather_than_align_by_luck():
    assert cap.paired_change(_verdicts("cc"), _verdicts("c")) is None


def test_nothing_comparable_returns_nothing():
    assert cap.paired_change(_verdicts("ii"), _verdicts("ii")) is None


# ── the report says what the numbers mean ────────────────────────────────────────────

def test_a_high_indeterminate_rate_is_called_out_as_a_budget_problem():
    """Because the accuracy above it is not a capability measurement when this is high."""
    lines = cap.report(cap.summarise(_verdicts("ci" * 10)))
    text = "\n".join(lines)
    assert "INDETERMINATE" in text
    assert "--max-new" in text


def test_an_interval_spanning_zero_is_said_out_loud():
    v = _verdicts("cw" * 20)
    change = cap.paired_change(v, list(v), resamples=200)
    text = "\n".join(cap.report(cap.summarise(v), change))
    assert "not distinguishable from no change" in text


def test_the_report_leads_with_counts():
    text = "\n".join(cap.report(cap.summarise(_verdicts("cccw"))))
    assert "graded 4 of 4" in text


# ── reading a graded benchmark ───────────────────────────────────────────────────────
# `resolve` returns one column, which is enough for a prompt set and not for anything marked.
# Pairing the question and the answer from two separate reads would be a place to misalign them
# silently, so they come back together or not at all.

def _jsonl(tmp_path, rows, name="b.jsonl"):
    import json
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_a_benchmark_reads_back_aligned(tmp_path):
    from senbonzakura import dataset
    p = _jsonl(tmp_path, [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(3)])
    q, a = dataset.resolve_pairs(p)
    assert q == ["q0", "q1", "q2"]
    assert a == ["#### 0", "#### 1", "#### 2"]


@pytest.mark.parametrize(("qcol", "acol"), [
    ("question", "answer"), ("problem", "solution"), ("query", "target"), ("prompt", "output"),
])
def test_the_usual_column_names_are_detected(tmp_path, qcol, acol):
    from senbonzakura import dataset
    p = _jsonl(tmp_path, [{qcol: "q", acol: "#### 1"}])
    assert dataset.resolve_pairs(p) == (["q"], ["#### 1"])


def test_an_explicit_column_name_wins(tmp_path):
    from senbonzakura import dataset
    p = _jsonl(tmp_path, [{"question": "wrong", "prompt": "right", "answer": "#### 1"}])
    q, _a = dataset.resolve_pairs(p, question_column="prompt")
    assert q == ["right"]


def test_a_named_column_that_does_not_exist_says_what_does(tmp_path):
    """Guessing which column is the answer is exactly the quiet decision that produces a confident
    wrong number, so it lists them instead.
    """
    from senbonzakura import dataset
    p = _jsonl(tmp_path, [{"question": "q", "answer": "#### 1"}])
    with pytest.raises(dataset.DatasetError, match="Available:"):
        dataset.resolve_pairs(p, answer_column="nope")


def test_a_file_with_no_recognisable_answer_column_refuses(tmp_path):
    from senbonzakura import dataset
    p = _jsonl(tmp_path, [{"question": "q", "notes": "x"}])
    with pytest.raises(dataset.DatasetError, match="answer"):
        dataset.resolve_pairs(p)


def test_the_same_column_cannot_be_both(tmp_path):
    """Every item would be marked against itself, which scores 100% and means nothing."""
    from senbonzakura import dataset
    p = _jsonl(tmp_path, [{"question": "q"}])
    with pytest.raises(dataset.DatasetError, match="same column"):
        dataset.resolve_pairs(p, question_column="question", answer_column="question")


def test_a_plain_text_file_is_refused_with_the_reason(tmp_path):
    from senbonzakura import dataset
    p = tmp_path / "plain.txt"
    p.write_text("just prompts\nand more\n", encoding="utf-8")
    with pytest.raises(dataset.DatasetError, match="two"):
        dataset.resolve_pairs(p)


def test_an_empty_benchmark_refuses(tmp_path):
    from senbonzakura import dataset
    p = _jsonl(tmp_path, [])
    with pytest.raises(dataset.DatasetError):
        dataset.resolve_pairs(p)


def test_a_missing_path_that_is_not_a_hub_id_refuses(tmp_path):
    from senbonzakura import dataset
    with pytest.raises(dataset.DatasetError, match="not a Hub id"):
        dataset.resolve_pairs(str(tmp_path / "no such file.jsonl"))


def test_a_slice_applies_to_both_sides_together(tmp_path):
    """Slicing one side and not the other would misalign every item after the cut."""
    from senbonzakura import dataset
    p = _jsonl(tmp_path, [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(10)])
    q, a = dataset.resolve_pairs(f"{p}::train[:3]")
    assert len(q) == len(a) == 3
    assert q[-1] == "q2" and a[-1] == "#### 2"


# ── generation reports its own truncation ────────────────────────────────────────────

class _FakeTok:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, texts, **_kw):
        import torch

        class _Enc(dict):
            """Mapping-shaped, because the real encoding is splatted into generate()."""

            def __init__(self, n):
                super().__init__(input_ids=torch.zeros(n, 3, dtype=torch.long))
                self.input_ids = self["input_ids"]

            def to(self, _device):
                return self
        return _Enc(len(texts))

    def decode(self, ids, **_kw):
        return " ".join(str(int(i)) for i in ids)

    def apply_chat_template(self, *_a, **_k):
        return "prompt"


class _FakeModel:
    """Emits one sequence that ends with EOS and one that does not."""

    def __init__(self, rows):
        self.rows = rows

    def generate(self, **_kw):
        import torch
        return torch.tensor(self.rows)


def test_generation_marks_only_the_sequence_that_ran_out_of_budget(monkeypatch):
    """An item is truncated when the budget ran out with no end-of-sequence token, and that is the
    one fact the grader cannot recover from the text.
    """
    from senbonzakura import capability, cli

    monkeypatch.setattr(cli, "render_chat", lambda _tok, p: p)
    #                      3 prompt tokens | new tokens
    rows = [[0, 0, 0, 9, 9, 2],          # ends with EOS -> finished
            [0, 0, 0, 9, 9, 9]]          # no EOS -> truncated
    gens, truncated = capability.generate_with_truncation(
        _FakeModel(rows), _FakeTok(), ["a", "b"], "cpu", batch=2, max_new=3)
    assert len(gens) == 2
    assert truncated == [False, True]


def test_a_tokenizer_with_no_eos_marks_everything_truncated(monkeypatch):
    """Without an end-of-sequence token there is no way to tell finished from cut off, and
    calling it finished would score truncations as wrong answers.
    """
    from senbonzakura import capability, cli

    monkeypatch.setattr(cli, "render_chat", lambda _tok, p: p)
    tok = _FakeTok()
    tok.eos_token_id = None
    _gens, truncated = capability.generate_with_truncation(
        _FakeModel([[0, 0, 0, 9, 9, 9]]), tok, ["a"], "cpu", batch=1, max_new=3)
    assert truncated == [True]


# ── the command's own guards ─────────────────────────────────────────────────────────

def test_a_reference_of_a_different_length_is_refused(tmp_path, monkeypatch):
    """Two arms compared on different item sets is not a paired comparison, and aligning them by
    position would produce a number that looks paired and is not.
    """
    import json

    from senbonzakura import capability

    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"verdicts": ["correct"] * 5}), encoding="utf-8")
    bench = _jsonl(tmp_path, [{"question": "q", "answer": "#### 1"}] * 2)
    with pytest.raises(SystemExit, match="same items in the same order"):
        capability.main(["--model", "m", "--device", "cpu", "--eval", str(bench),
                         "--n", "2", "--out", str(tmp_path / "o.json"),
                         "--compare-to", str(ref)])


def test_skipping_past_the_end_is_refused(tmp_path):
    from senbonzakura import capability
    bench = _jsonl(tmp_path, [{"question": "q", "answer": "#### 1"}])
    with pytest.raises(SystemExit, match="leaves nothing"):
        capability.main(["--model", "m", "--device", "cpu", "--eval", str(bench),
                         "--skip", "5", "--out", str(tmp_path / "o.json")])


def test_asking_for_more_items_than_exist_is_refused(tmp_path):
    from senbonzakura import capability
    bench = _jsonl(tmp_path, [{"question": "q", "answer": "#### 1"}])
    with pytest.raises(SystemExit, match="exceeds"):
        capability.main(["--model", "m", "--device", "cpu", "--eval", str(bench),
                         "--n", "50", "--out", str(tmp_path / "o.json")])


def test_a_reference_file_is_read_back_as_verdicts(tmp_path):
    import json

    from senbonzakura import capability

    p = tmp_path / "r.json"
    p.write_text(json.dumps({"verdicts": ["correct", "wrong"]}), encoding="utf-8")
    assert capability.load_reference(p) == ["correct", "wrong"]
    assert capability.load_reference("") is None


def test_the_command_runs_end_to_end_and_reports_nothing_gradeable(tmp_path, monkeypatch,
                                                                   tiny_model, tiny_tok):
    """The whole path, on a model that cannot do arithmetic.

    A model with untrained weights produces no numbers, so every item is indeterminate. What is
    under test is that the command SAYS so: "nothing could be graded" rather than an accuracy of
    zero, an indeterminate rate of 100%, and a non-zero exit. A run that measured nothing must not
    be mistakable for a run that measured a zero, and that is the property this whole module is
    built around.

    Smoke-tested against a real Hub model on CPU when it was written; this is the same path with
    the in-repo fixtures so it costs no network.
    """
    import json

    from senbonzakura import capability

    monkeypatch.setattr(capability, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok),
                        raising=False)
    import senbonzakura.cli as _cli
    monkeypatch.setattr(_cli, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    monkeypatch.setattr(_cli, "render_chat", lambda _tok, p: p)

    bench = _jsonl(tmp_path, [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(3)])
    out = tmp_path / "cap.json"
    gens = tmp_path / "gens.jsonl"
    code = capability.main(["--model", "m", "--device", "cpu", "--eval", str(bench),
                            "--n", "3", "--max-new", "4", "--batch", "2",
                            "--out", str(out), "--save-generations", str(gens)])

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["summary"]["n"] == 3
    assert doc["summary"]["accuracy"] is None, "an ungradeable run must not report an accuracy"
    assert doc["summary"]["indeterminate"] == 3
    assert code != 0, "a run that graded nothing must not exit as though it succeeded"
    assert len(gens.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_a_nonsense_item_count_is_refused(tmp_path):
    from senbonzakura import capability
    bench = _jsonl(tmp_path, [{"question": "q", "answer": "#### 1"}])
    with pytest.raises(SystemExit, match="at least 1"):
        capability.main(["--model", "m", "--eval", str(bench), "--n", "0",
                         "--out", str(tmp_path / "o.json")])


def test_an_unreadable_benchmark_is_refused_with_the_readers_reason(tmp_path):
    from senbonzakura import capability
    with pytest.raises(SystemExit, match="not a Hub id"):
        capability.main(["--model", "m", "--eval", str(tmp_path / "no such file.jsonl"),
                         "--out", str(tmp_path / "o.json")])


def test_the_paired_change_reaches_the_output(tmp_path, monkeypatch, tiny_model, tiny_tok):
    """The comparison is the number the whole command exists for, so it has to survive the round
    trip into the artefact rather than only being printed.
    """
    import json

    import senbonzakura.cli as _cli
    from senbonzakura import capability

    monkeypatch.setattr(_cli, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    monkeypatch.setattr(_cli, "render_chat", lambda _tok, p: p)
    # A reference whose items are all gradeable, against a run whose items are all indeterminate:
    # nothing is comparable, so the change must be absent rather than fabricated as zero.
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"verdicts": ["correct", "wrong"]}), encoding="utf-8")
    bench = _jsonl(tmp_path, [{"question": "q", "answer": "#### 1"}] * 2)
    out = tmp_path / "cap.json"
    capability.main(["--model", "m", "--device", "cpu", "--eval", str(bench), "--n", "2",
                     "--max-new", "4", "--out", str(out), "--compare-to", str(ref)])
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["compare_to"] == str(ref)
    assert doc["change"] is None, "no comparable pairs must not become a change of zero"


# ── more than one task (B1) ──────────────────────────────────────────────────────────
# Arithmetic is one capability. A model can keep it and lose instruction-following, or lose the
# ability to pick the right option from a list, and a single benchmark reports none of that.
# Every task is graded by code against a reference: no judge, so nothing whose reliability would
# have to be established before the result meant anything.

def test_every_task_says_what_it_grades_and_why_no_judge_is_needed():
    """A task that cannot answer both is a benchmark somebody has to take on trust."""
    for name in cap.TASK_CHOICES:
        t = cap.get_task(name)
        assert t.grades and t.needs_no_judge, f"{name} does not explain itself"
        assert "{}" in t.prompt, f"{name} has no place to put the question"


def test_an_unknown_task_lists_what_exists():
    with pytest.raises(KeyError, match="Available:"):
        cap.get_task("vibes")


def test_the_default_task_is_unchanged():
    """Changing it would silently change what every existing capability run measured."""
    assert cap.DEFAULT_TASK == "numeric"


@pytest.mark.parametrize(("text", "want"), [
    ("(C)", "C"),
    ("The answer is B.", "B"),
    ("A) wrong, B) also wrong, so C", "C"),      # last, not first
    ("D:", "D"),
    ("no letters here", None),
    ("", None),
    (None, None),
])
def test_a_multiple_choice_answer_is_read_from_prose(text, want):
    assert cap.choice_answer(text) == want


def test_a_choice_is_not_found_in_an_ordinary_word():
    """Anchoring matters: without it, a capital inside a word becomes an answer."""
    assert cap.choice_answer("ABLATION removes a direction") != "B"


@pytest.mark.parametrize(("got", "gold", "want"), [
    ("Paris", "paris", "correct"),
    ("The Paris", "paris", "correct"),           # article and case normalised
    ("paris.", "paris", "correct"),              # trailing punctuation
    ("Lyon", "paris", "wrong"),
    ("", "paris", "indeterminate"),
])
def test_the_exact_task_normalises_only_what_is_not_the_answer(got, gold, want):
    assert cap.grade_one(got, gold, task="exact") == want


def test_the_exact_task_takes_the_last_line_because_that_is_what_it_asked_for():
    """The prompt says "on the last line and nothing else", so grading anything else would be
    marking against a different instruction than the one given.
    """
    assert cap.grade_one("Let me think.\nWorking.\nParis", "Paris", task="exact") == "correct"


def test_the_three_rules_hold_on_every_task():
    """Truncation is indeterminate, an unusable row is indeterminate, and a refusal is neither
    correct nor wrong. These are the properties the whole module exists for, so they cannot hold
    on the first task and quietly lapse on the others.
    """
    for name in cap.TASK_CHOICES:
        assert cap.grade_one("anything", "1", truncated=True, task=name) == "indeterminate"
        assert cap.grade_one("anything", None, task=name) == "indeterminate"
        assert cap.grade_one("I'm sorry, I can't help with that.", "#### 8",
                             task="numeric") == "indeterminate"


def test_the_task_reaches_the_output_because_an_accuracy_needs_to_say_what_was_asked(
        tmp_path, monkeypatch, tiny_model, tiny_tok):
    import json

    import senbonzakura.cli as _cli

    monkeypatch.setattr(_cli, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    monkeypatch.setattr(_cli, "render_chat", lambda _tok, p: p)
    bench = _jsonl(tmp_path, [{"question": "q", "answer": "B"}] * 2)
    out = tmp_path / "c.json"
    cap.main(["--model", "m", "--device", "cpu", "--eval", str(bench), "--n", "2",
              "--max-new", "4", "--task", "multiple-choice", "--out", str(out)])
    assert json.loads(out.read_text(encoding="utf-8"))["task"] == "multiple-choice"


# ── agentic behaviour, graded without a judge (B2) ───────────────────────────────────
# "Agentic" sounds like the thing that finally forces a judge, and most of it does not. Whether a
# model emitted a well-formed call, named the right tool, passed the right arguments and obeyed a
# format instruction are all questions code answers exactly. Reaching for a judge early would make
# every number inherit its reliability, which is what the judge harness exists to stop.

@pytest.mark.parametrize("text", [
    '{"name": "search", "arguments": {"q": "cats"}}',
    'Sure.\n```json\n{"name": "search", "arguments": {"q": "cats"}}\n```',
    'I will call it: {"name": "search", "arguments": {"q": "cats"}} now.',
    '{"tool": "search", "args": {"q": "cats"}}',
    '{"name": "search", "arguments": "{\\"q\\": \\"cats\\"}"}',   # args as a JSON string
])
def test_a_tool_call_is_found_in_the_shapes_models_actually_emit(text):
    assert cap.tool_call(text) == ("search", {"q": "cats"})


def test_a_nested_function_name_is_understood():
    assert cap.tool_call('{"function": {"name": "search"}, "arguments": {}}')[0] == "search"


def test_argument_order_is_not_part_of_the_answer():
    """Two calls differing only in key order are the same call, and marking them different would
    report a formatting change as a capability loss.
    """
    a = cap.tool_call('{"name": "f", "arguments": {"x": 1, "y": 2}}')
    b = cap.tool_call('{"name": "f", "arguments": {"y": 2, "x": 1}}')
    assert cap.same_tool_call(a, b)


def test_the_wrong_tool_is_wrong():
    assert cap.grade_one('{"name": "delete", "arguments": {}}',
                         '{"name": "search", "arguments": {}}', task="tool-call") == "wrong"


def test_the_right_tool_with_wrong_arguments_is_wrong():
    """Naming the tool and then passing the wrong thing to it is the failure that matters most in
    an agent, and a name-only comparison would score it correct.
    """
    assert cap.grade_one('{"name": "search", "arguments": {"q": "dogs"}}',
                         '{"name": "search", "arguments": {"q": "cats"}}',
                         task="tool-call") == "wrong"


def test_replying_in_prose_when_asked_for_a_call_counts_as_wrong():
    """THE DISTINCTION THIS TASK TURNS ON.

    Elsewhere an unreadable answer is indeterminate, because a model that says "eight" in words
    did the arithmetic and formatted it unexpectedly. Here producing the format IS the capability,
    so a prose reply has failed the thing being measured, and calling it indeterminate would hide
    the most common way tool use breaks.
    """
    assert cap.grade_one("I would search for cats.",
                         '{"name": "search", "arguments": {}}', task="tool-call") == "wrong"


def test_a_truncated_tool_call_is_still_indeterminate():
    """Truncation is decided before any of that, so a cut-off generation is never scored wrong."""
    assert cap.grade_one("I would sea", '{"name": "search", "arguments": {}}',
                         truncated=True, task="tool-call") == "indeterminate"


def test_a_model_answering_arithmetic_in_words_is_still_indeterminate():
    """The contrast that makes the rule above a distinction rather than an inconsistency."""
    assert cap.grade_one("the answer is eight", "#### 8") == "indeterminate"


# ── format instructions ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("text", "spec", "want"), [
    ("hello there", "max_words:3", True),
    ("one two three four", "max_words:3", False),
    ("hello", "min_words:1;lowercase", True),
    ("HELLO", "lowercase", False),
    ("HELLO", "uppercase", True),
    ("a\nb\nc", "lines:3", True),
    ("say cats please", "contains:cats", True),
    ("say dogs please", "excludes:cats", True),
    ("say cats please", "excludes:cats", False),
    ('{"a": 1}', "json", True),
    ("not json at all", "json", False),
    ("finish here.", "ends_with:.", True),
    ("Answer: yes", "starts_with:Answer", True),
])
def test_each_constraint_is_a_rule_a_reader_could_apply_by_hand(text, spec, want):
    checks = cap.parse_constraints(spec)
    assert cap.meets_constraints(text, checks) is want


def test_all_the_constraints_must_hold_not_most_of_them():
    """A response that obeys three instructions and breaks the fourth has not followed the
    instruction.
    """
    checks = cap.parse_constraints("max_words:5;lowercase;contains:cat")
    assert cap.meets_constraints("i like my cat", checks) is True
    assert cap.meets_constraints("I like my cat", checks) is False


def test_an_unknown_constraint_is_refused_rather_than_skipped():
    """A skipped check is one the model is graded as having passed, so a typo in a benchmark would
    quietly make every item easier.
    """
    with pytest.raises(KeyError, match="Available:"):
        cap.parse_constraints("max_words:5;vibes:good")


def test_an_empty_constraint_spec_is_an_unusable_row():
    assert cap.parse_constraints("") is None
    assert cap.grade_one("anything", "", task="constraints") == "indeterminate"


def test_every_task_still_explains_itself():
    """Including the two new ones: a task that cannot say what it grades and why no judge is
    needed is a benchmark somebody has to take on trust.
    """
    for name in cap.TASK_CHOICES:
        t = cap.get_task(name)
        assert t.grades and t.needs_no_judge, f"{name} does not explain itself"
