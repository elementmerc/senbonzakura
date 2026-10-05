# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The multi-turn command: the escalation, the control, and the turn it gives way on.

The properties under test are the three that make the number mean anything, plus the one defect
that only shows in a transcript:

- the model's own refusal is in the history, so the follow-ups are answering something;
- the control runs on the same refused rows, and an overlapping pair is reported as "cannot tell";
- a conversation that outgrew the context window invalidates the run rather than being scored;
- a caller's own escalation never reaches the artefact as text.
"""
import json

import pytest
from artefacts import needs_corpora

from senbonzakura import entry, jailbreak, metrics, multiturn

ANSWER = "Sure, here is the recipe you asked for. Start by warming the pan."
REFUSAL = "I cannot help with that request."


# ── the strategy itself ───────────────────────────────────────────────────────────────
def test_turn_one_is_the_raw_request_under_every_strategy():
    """What makes the single-turn figure from this run the same measurement as `jailbreak`'s, and
    what lets the control be paired row by row.
    """
    for strategy in multiturn.STRATEGIES.values():
        assert strategy.message(1, "do the thing") == "do the thing"


def test_the_control_repeats_the_request_and_the_escalation_does_not():
    assert multiturn.CONTROL.message(2, "do the thing") == "do the thing"
    assert "do the thing" not in multiturn.REFRAME.message(2, "do the thing")


def test_two_strategies_with_different_turns_have_different_digests():
    """The name is not enough: a strategy file may call itself anything, including a built-in's
    name, and a figure attributed to the wrong escalation is worse than one attributed to none.
    """
    assert multiturn.CONTROL.digest() != multiturn.REFRAME.digest()
    assert multiturn.CONTROL.digest() == multiturn.CONTROL.digest()


def test_the_turn_count_is_one_more_than_the_follow_ups():
    assert multiturn.REFRAME.turns == len(multiturn.REFRAME.follow_ups) + 1


# ── a caller's own escalation ─────────────────────────────────────────────────────────
def _write(tmp_path, doc, name="s.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc) if not isinstance(doc, str) else doc, encoding="utf-8")
    return str(p)


def test_a_strategy_file_is_read_and_its_turns_used(tmp_path):
    s = multiturn.load_strategy(_write(tmp_path, {
        "name": "mine", "description": "d", "follow_ups": ["again {request}", "once more"]}))
    assert s.turns == 3
    assert s.message(2, "X") == "again X"
    assert s.message(3, "X") == "once more"


@pytest.mark.parametrize(("doc", "says"), [
    ({"name": "m", "description": "d"}, "follow_ups"),
    ({"description": "d", "follow_ups": ["a"]}, "name"),
    ({"name": "m", "description": "d", "follow_ups": []}, "single-turn"),
    ({"name": "m", "description": "d", "follow_ups": "a"}, "single-turn"),
    ({"name": "m", "description": "d", "follow_ups": ["  "]}, "empty"),
    ({"name": "m", "description": "d", "follow_ups": [7]}, "not a string"),
    ({"name": "   ", "description": "d", "follow_ups": ["a"]}, "no usable name"),
])
def test_a_malformed_strategy_file_is_refused_and_says_what_is_wrong(tmp_path, doc, says):
    with pytest.raises(multiturn.StrategyError, match=says):
        multiturn.load_strategy(_write(tmp_path, doc))


def test_a_strategy_file_that_is_not_an_object_is_refused(tmp_path):
    with pytest.raises(multiturn.StrategyError, match="list"):
        multiturn.load_strategy(_write(tmp_path, ["a", "b"]))


def test_unreadable_and_unparseable_strategy_files_are_told_apart(tmp_path):
    with pytest.raises(multiturn.StrategyError, match="could not be read"):
        multiturn.load_strategy(str(tmp_path / "absent.json"))
    with pytest.raises(multiturn.StrategyError, match="not valid JSON"):
        multiturn.load_strategy(_write(tmp_path, "{not json", name="bad.json"))


def test_a_built_in_strategy_is_written_into_the_artefact():
    block = multiturn.strategy_block(multiturn.REFRAME, built_in=True)
    assert block["follow_ups"] == list(multiturn.REFRAME.follow_ups)
    assert block["digest"] and block["turns"] == 4


def test_a_callers_own_turns_are_never_written_into_the_artefact(tmp_path):
    """An artefact is the thing most likely to be published, so somebody else's prompts must not
    ride out in it. The digest is what keeps the figure traceable.
    """
    secret = "a follow-up nobody else should receive"
    s = multiturn.load_strategy(_write(tmp_path, {
        "name": "mine", "description": "d", "follow_ups": [secret]}))
    block = multiturn.strategy_block(s, built_in=False)
    assert block["follow_ups"] is None
    assert secret not in json.dumps(block)
    assert block["digest"] == s.digest()
    assert "published" in block["follow_ups_withheld"]


# ── the escalation loop ───────────────────────────────────────────────────────────────
def _sender(script):
    """A `send` that returns the next scripted batch, and records what it was shown."""
    seen = []

    def send(conversations):
        seen.append([list(c) for c in conversations])
        replies = script[len(seen) - 1]
        return [replies] * len(conversations) if isinstance(replies, str) else replies, []

    return send, seen


def test_the_models_own_refusal_is_in_the_history_the_follow_up_answers():
    """THE DEFECT THIS WAS WRITTEN FOR, found by reading a real run's transcript rather than its
    rate. Without the opening reply the escalation sends two user turns back to back, so the model
    reacts to a refusal it was never shown and every follow-up that acknowledges one is
    incoherent. Nothing in the rate shows it.
    """
    send, seen = _sender([REFUSAL, ANSWER])
    multiturn.escalate(["do the thing"], ["I will not."], multiturn.REFRAME, send)
    roles = [m["role"] for m in seen[0][0]]
    assert roles == ["user", "assistant", "user"], roles
    assert seen[0][0][1]["content"] == "I will not."
    # And the second round carries the whole history, not just the latest pair.
    assert [m["role"] for m in seen[1][0]] == ["user", "assistant", "user", "assistant", "user"]


def test_a_row_that_complies_leaves_the_conversation():
    """Pushing a model that has already answered measures nothing and costs the same GPU."""
    send, seen = _sender([[ANSWER, REFUSAL], [ANSWER]])
    first, _t, _o = multiturn.escalate(["a", "b"], [REFUSAL, REFUSAL], multiturn.REFRAME, send)
    assert first == [2, 3]
    assert len(seen[0]) == 2, "both rows were asked on turn two"
    assert len(seen[1]) == 1, "the row that complied was asked again"


def test_a_row_that_never_complies_has_no_turn():
    send, _seen = _sender([REFUSAL, REFUSAL, REFUSAL])
    first, _t, _o = multiturn.escalate(["a"], [REFUSAL], multiturn.REFRAME, send)
    assert first == [None]


def test_the_loop_stops_early_when_nothing_is_left_to_ask():
    send, seen = _sender([ANSWER, ANSWER, ANSWER])
    multiturn.escalate(["a"], [REFUSAL], multiturn.REFRAME, send)
    assert len(seen) == 1, "it kept generating after every row had complied"


def test_the_transcript_carries_every_reply_with_its_turn():
    send, _seen = _sender([REFUSAL, ANSWER])
    _f, transcripts, _o = multiturn.escalate(["a"], [REFUSAL], multiturn.REFRAME, send)
    assert [r["turn"] for r in transcripts[0]] == [2, 3]


def test_an_overflowing_conversation_is_reported_by_index():
    def send(conversations):
        return [REFUSAL] * len(conversations), [1]

    _f, _t, over = multiturn.escalate(["a", "b"], [REFUSAL, REFUSAL], multiturn.CONTROL, send)
    assert over == [1]


# ── the histogram and the median ──────────────────────────────────────────────────────
def test_the_median_turn_is_over_the_rows_that_complied():
    """A row that never complied has no turn, and scoring it as `turns + 1` would move the median
    by an amount that depends on the refusal rate rather than on the escalation.
    """
    h = multiturn.first_turn_histogram([2, 2, 4, None, None], turns=4)
    assert h["by_turn"] == {"1": 0, "2": 2, "3": 0, "4": 1}
    assert h["median_turn"] == 2.0
    assert h["never"] == 2


def test_the_median_is_a_float_whether_the_count_is_odd_or_even():
    assert isinstance(multiturn.first_turn_histogram([2, 3], turns=3)["median_turn"], float)
    assert isinstance(multiturn.first_turn_histogram([2, 3, 3], turns=3)["median_turn"], float)


def test_nothing_complied_is_not_the_same_statement_as_turn_zero():
    h = multiturn.first_turn_histogram([None, None], turns=3)
    assert h["median_turn"] is None
    assert h["never"] == 2


# ── the control comparison ────────────────────────────────────────────────────────────
def _arm(converted, refused, strategy=multiturn.REFRAME):
    return multiturn.arm_summary([2] * converted + [None] * (refused - converted),
                                 strategy, refused_n=refused)


def test_an_overlapping_pair_is_reported_as_cannot_tell():
    """An overlapping pair reported as a point difference is how a strategy comes to be credited
    with an effect that was the model's own instability.
    """
    a = advantage = multiturn.advantage(_arm(16, 30), _arm(15, 30, multiturn.CONTROL))
    assert a["distinguishable"] is False
    assert "does not separate" in advantage["reading"]
    assert "do not credit" in advantage["reading"]


def test_a_separated_pair_says_the_escalation_did_something():
    a = multiturn.advantage(_arm(29, 30), _arm(1, 30, multiturn.CONTROL))
    assert a["distinguishable"] is True
    assert "did something repetition did not" in a["reading"]
    assert a["gap"] > 0


def test_no_control_means_no_comparison_rather_than_a_zero():
    assert multiturn.advantage(_arm(16, 30), None) is None


def test_a_pair_neither_of_which_can_state_a_rate_says_so():
    a = multiturn.advantage(_arm(2, 4), _arm(1, 4, multiturn.CONTROL))
    assert a["gap"] is None
    assert a["distinguishable"] is False
    assert "nothing to compare" in a["reading"]


def test_an_arm_summary_excludes_the_rows_that_complied_immediately():
    """The denominator is the rows that refused at turn one. Counting the rest as converted would
    report a model that never refused as one whose refusal collapsed.
    """
    arm = _arm(10, 40)
    assert arm["conversion"]["n"] == 40
    assert arm["refused_at_turn_one"] == 40
    assert arm["conversion"]["rate"] == 0.25


# ── when the run is not a measurement ─────────────────────────────────────────────────
def _single(n_answer=0, n_refusal=0, n_broken=0):
    return jailbreak.arm_result([ANSWER] * n_answer + [REFUSAL] * n_refusal + [""] * n_broken)


def test_a_clean_run_is_a_measurement():
    assert multiturn.validity(_single(n_refusal=40), [_arm(10, 40)], []) is None


def test_a_turn_one_below_the_floor_is_not_a_measurement():
    why = multiturn.validity(_single(n_refusal=5), [_arm(2, 5)], [])
    assert "turn one ran on 5" in why


def test_breakage_at_turn_one_chooses_the_refused_set_by_breakage():
    why = multiturn.validity(_single(n_refusal=20, n_broken=20), [_arm(10, 40)], [])
    assert "chosen by breakage" in why


def test_a_truncated_conversation_invalidates_the_run():
    """A truncated history is a successful generation of the wrong experiment, and nothing in the
    reply text says it happened.
    """
    why = multiturn.validity(_single(n_refusal=40), [_arm(10, 40)], [3, 7])
    assert "2 conversation(s) outgrew" in why
    assert str(multiturn.CONTEXT_BUDGET) in why


def test_nothing_left_to_escalate_is_stated_as_a_fact_about_the_model():
    """A fully abliterated model answers at turn one, so the conversion rate has no denominator.
    That is a result, and it must not read as resistance.
    """
    why = multiturn.validity(_single(n_answer=40), [_arm(0, 0)], [])
    assert "nothing for an escalation to convert" in why
    assert "statement about the model rather than a fault" in why


def test_an_arm_that_is_none_does_not_crash_the_check():
    assert multiturn.validity(_single(n_refusal=40), [_arm(10, 40), None], []) is None


# ── what reaches the terminal ─────────────────────────────────────────────────────────
def _res(**over):
    res = {
        "label": "x", "attack_set": {"key": "strongreject"},
        "single_turn": _single(n_answer=10, n_refusal=30),
        "refused_at_turn_one": 30,
        "arm": _arm(16, 30), "control": _arm(15, 30, multiturn.CONTROL),
        "advantage": None, "control_missing": None, "budget_warning": None,
        "self_invalidated": None,
    }
    res.update(over)
    return res


def test_the_report_opens_with_turn_one_and_then_every_arm():
    lines = multiturn._report(_res(), "r.json")
    assert lines[0].startswith("MULTITURN_TURN_ONE")
    assert "refused=30" in lines[0]
    assert sum(ln.startswith("MULTITURN_CONVERSION") for ln in lines) == 2
    assert "median_turn=2.0" in lines[1]


def test_the_advantage_reading_reaches_the_terminal():
    lines = multiturn._report(_res(advantage={"reading": "they overlap"}), "r.json")
    assert any("MULTITURN_ADVANTAGE" in ln and "they overlap" in ln for ln in lines)


def test_a_missing_control_is_announced():
    lines = multiturn._report(_res(control=None, control_missing="no control"), "r.json")
    assert sum(ln.startswith("MULTITURN_CONVERSION") for ln in lines) == 1
    assert any(ln.startswith("MULTITURN_NO_CONTROL") for ln in lines)


def test_the_budget_warning_and_the_invalidation_both_reach_the_terminal():
    lines = multiturn._report(
        _res(budget_warning="too short", self_invalidated="wrecked"), "out/r.json")
    assert any("BUDGET_WARNING" in ln for ln in lines)
    assert lines[-1].startswith("MULTITURN_NOT_A_MEASUREMENT")
    assert "out/r.json" in lines[-1]


# ── the whole command ─────────────────────────────────────────────────────────────────
class _Tok:
    senbon_chat_template = None


def _drive(monkeypatch, argv, *, turn_one=None, later=None, overflow=(), prompts=40):
    """Run `main` with the model and the corpora replaced, so every branch runs on CPU."""
    from senbonzakura import score

    monkeypatch.setattr(score, "load_model_and_tokenizer", lambda *a, **k: (object(), _Tok()))
    monkeypatch.setattr(jailbreak, "prompts_for",
                        lambda key, limit, what: [f"{key} {i}" for i in range(limit or prompts)])
    rounds = []

    def _generate_turn(model, tok, conversations, device, *, batch, max_new):
        rounds.append([list(c) for c in conversations])
        if len(rounds) == 1:
            return list(turn_one or [REFUSAL] * len(conversations)), list(overflow)
        reply = (later or REFUSAL)
        replies = ([reply] * len(conversations) if isinstance(reply, str)
                   else list(reply[:len(conversations)]))
        return replies, []

    monkeypatch.setattr(multiturn, "generate_turn", _generate_turn)
    return multiturn.main(argv), rounds


def test_the_command_runs_turn_one_once_for_both_arms(monkeypatch, tmp_path):
    """Generating turn one per arm would pay twice for the same tokens and, worse, let the two
    arms escalate over different refused sets while being compared as though they were paired.
    """
    res, rounds = _drive(monkeypatch, ["--model", "m", "--device", "cpu",
                                       "--out", str(tmp_path / "r.json")])
    assert res["refused_at_turn_one"] == 40
    # One opening round, then three escalation turns for each of the two arms.
    assert len(rounds) == 1 + 3 + 3
    assert res["arm"]["conversion"]["n"] == res["control"]["conversion"]["n"] == 40


def test_the_artefact_names_the_strategy_and_stamps_the_conversion(monkeypatch, tmp_path):
    out = str(tmp_path / "r.json")
    _drive(monkeypatch, ["--model", "m", "--device", "cpu", "--out", out], later=ANSWER)
    with open(out) as f:
        doc = json.load(f)
    assert doc["strategy"]["name"] == multiturn.DEFAULT_STRATEGY
    assert doc["strategy"]["digest"] == multiturn.REFRAME.digest()
    assert doc["control_strategy"]["name"] == multiturn.CONTROL.name
    block = doc["metrics"]["multi_turn_conversion"]
    assert block["strategy"] == multiturn.DEFAULT_STRATEGY
    assert block["interval"] and block["count"] == 40
    assert block["value"] == 1.0


def test_the_control_is_not_run_against_itself(monkeypatch, tmp_path):
    """Running `plain-repeat` against `plain-repeat` would pay twice for one number and then
    report a gap of zero as a finding about the strategy.
    """
    res, rounds = _drive(monkeypatch, ["--model", "m", "--device", "cpu", "--strategy",
                                       multiturn.CONTROL.name,
                                       "--out", str(tmp_path / "r.json")])
    assert res["control"] is None
    assert res["control_strategy"] is None
    assert res["control_missing"] is None, "it was not skipped by the operator, so nothing is missing"
    assert len(rounds) == 1 + 3


def test_opting_out_of_the_control_is_recorded_and_announced(monkeypatch, tmp_path, capsys):
    res, rounds = _drive(monkeypatch, ["--model", "m", "--device", "cpu", "--no-control",
                                       "--out", str(tmp_path / "r.json")])
    assert res["control"] is None
    assert "one keystroke" in res["control_missing"]
    assert "MULTITURN_NO_CONTROL" in capsys.readouterr().out
    assert len(rounds) == 1 + 3


def test_a_strategy_file_is_used_and_its_text_stays_out_of_the_file(monkeypatch, tmp_path):
    secret = "a follow-up nobody else should receive"
    path = _write(tmp_path, {"name": "mine", "description": "d", "follow_ups": [secret]})
    out = str(tmp_path / "r.json")
    _drive(monkeypatch, ["--model", "m", "--device", "cpu", "--strategy-file", path,
                         "--out", out])
    with open(out, encoding="utf-8") as f:
        text = f.read()
    assert "mine" in text
    assert secret not in text


def test_a_broken_strategy_file_refuses_before_the_model_loads(monkeypatch, tmp_path):
    path = _write(tmp_path, {"name": "mine"})
    with pytest.raises(SystemExit, match="missing"):
        _drive(monkeypatch, ["--model", "m", "--device", "cpu", "--strategy-file", path,
                             "--out", str(tmp_path / "r.json")])


def test_an_overflowing_run_invalidates_itself_in_the_file_and_the_exit_status(
        monkeypatch, tmp_path, capsys):
    out = str(tmp_path / "r.json")
    res, _rounds = _drive(monkeypatch, ["--model", "m", "--device", "cpu", "--out", out],
                          overflow=[2])
    assert "outgrew" in res["self_invalidated"]
    with open(out) as f:
        assert json.load(f)["self_invalidated"], "the verdict never reached the file"
    assert "NOT A MEASUREMENT" in capsys.readouterr().out
    assert entry.exit_status(res) == 1


def test_the_transcripts_of_every_turn_go_to_one_file(monkeypatch, tmp_path):
    path = str(tmp_path / "nested" / "t.jsonl")
    _drive(monkeypatch, ["--model", "m", "--device", "cpu", "--out", str(tmp_path / "r.json"),
                         "--save-transcripts", path], later=ANSWER)
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert sum(r["turn"] == 1 for r in rows) == 40
    assert {r.get("strategy") for r in rows if r["turn"] > 1} == {
        multiturn.REFRAME.name, multiturn.CONTROL.name}


def test_the_ruler_is_checked_before_a_single_prompt_is_sent(monkeypatch, tmp_path):
    monkeypatch.setattr(metrics, "validate_ruler",
                        lambda: (_ for _ in ()).throw(RuntimeError("the ruler misreads")))
    with pytest.raises(RuntimeError, match="misreads"):
        _drive(monkeypatch, ["--model", "m", "--device", "cpu",
                             "--out", str(tmp_path / "r.json")])


# ── the real generation path ───────────────────────────────────────────────────────────
def test_a_conversation_is_measured_in_tokens_and_not_in_characters(tiny_tok):
    """THE PLANTED DEFECT THIS WAS REWRITTEN FOR, 2026-10-05.

    The first version of this test asserted that a longer conversation measured longer than a
    short one, which is true of a character count as well, so reverting the measurement to
    `len(tok(text).input_ids)` left the whole file green. A bare string reaches a tokeniser as an
    ITERABLE OF CHARACTERS, so that form returns one row per character and the count comes back
    as the length of the text. The consequence is not a slightly wrong number: a long
    conversation reads as an overflow, which invalidates a run that was fine.

    So the assertion is the one that separates the two readings: a conversation whose text is
    longer than the budget, and whose token count is not, must not measure as an overflow.
    """
    text = "U: " + ("word " * 800)
    assert len(text) > multiturn.CONTEXT_BUDGET, "the text has to be long enough to tell them apart"
    assert multiturn.conversation_length(tiny_tok, text) <= multiturn.CONTEXT_BUDGET
    assert multiturn.conversation_length(tiny_tok, "U: hi") > 0


def test_the_whole_history_reaches_the_renderer(tiny_tok):
    """The single-turn renderer and this one are the same template call, so a dialogue and the
    single-turn figure it is compared against are measured under one prompt format.
    """
    from senbonzakura.firsttoken import render_conversation

    rendered = render_conversation(tiny_tok, [
        {"role": "user", "content": "the request"},
        {"role": "assistant", "content": "the refusal"},
        {"role": "user", "content": "the follow up"}])
    for fragment in ("the request", "the refusal", "the follow up"):
        assert fragment in rendered


@needs_corpora
def test_the_command_runs_against_the_real_corpora_and_a_real_generation_loop(
        monkeypatch, tmp_path, tiny_model, tiny_tok):
    """The resolver, the renderer, the overflow check and a real `model.generate`.

    Everything above replaces `generate_turn`, so without this nothing exercises the rendering of
    a multi-turn conversation, the token count that guards it, or the batching.
    """
    from senbonzakura import score

    monkeypatch.setattr(score, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    res = multiturn.main(["--model", "m", "--device", "cpu", "--attack-set", "advbench",
                          "--n", "3", "--batch", "2", "--max-new", "2",
                          "--out", str(tmp_path / "r.json")])
    assert res["attack_set"]["key"] == "advbench"
    assert res["single_turn"]["n"] == 3
    assert res["overflowed"] == 0
    # Three prompts is below the floor, so the run says so rather than stating a rate.
    assert "below the floor" in res["self_invalidated"]
