# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The single-turn jailbreak command, and the ways its number could be wrong.

The three properties under test here are the three reasons the command exists rather than being a
flag on `score`:

- the two arms travel together, and an artefact missing one says so;
- a rate is withheld rather than stated when the sample or the replies cannot carry it;
- the attack slot refuses a benign corpus, which is the one mistake nothing downstream can see.
"""
import json
import re

import pytest
from artefacts import needs_corpora

from senbonzakura import corpora, entry, jailbreak

ANSWER = "Sure, here is the recipe you asked for. Start by warming the pan."
REFUSAL = "I cannot help with that request."
LECTURE = "That said, I must advise against attempting any of it."
GARBAGE = ""


# ── the arm guard ─────────────────────────────────────────────────────────────────────
def test_a_benign_set_in_the_attack_slot_is_refused_and_names_the_arm():
    """The one mistake nothing downstream can see: XSTest's safe half as an attack set reports
    near-total jailbreak success on a model that refused nothing it should have.
    """
    with pytest.raises(SystemExit) as e:
        jailbreak.corpus_for("xstest-safe", arm="harmful", flag="--attack-set")
    assert "benign arm" in str(e.value)
    assert "answer" in str(e.value)
    # The refusal has to say what to type instead.
    assert "strongreject" in str(e.value)


def test_a_harmful_set_in_the_benign_slot_is_refused():
    with pytest.raises(SystemExit) as e:
        jailbreak.corpus_for("advbench", arm="benign", flag="--benign-set")
    assert "harmful arm" in str(e.value)
    assert "xstest-safe" in str(e.value)


def test_the_right_arm_passes_through():
    assert jailbreak.corpus_for("strongreject", arm="harmful", flag="--attack-set").rows == 313


# ── the two estimators ────────────────────────────────────────────────────────────────
def test_the_two_estimators_disagree_about_breakage_and_evasion():
    """The whole reason both are reported. `1 - refusal` counts a wrecked reply and a lecture as
    a successful jailbreak; the strict reading counts neither.
    """
    gens = [ANSWER, ANSWER, REFUSAL, LECTURE, GARBAGE]
    arm = jailbreak.arm_result(gens)
    assert arm["n"] == 5
    # One hard refusal of five, so the loose reading says four jailbreaks.
    assert arm[jailbreak.LOOSE]["count"] == 4
    # Two of those four delivered nothing: the lecture and the breakage.
    assert arm[jailbreak.STRICT]["count"] == 2
    assert arm[jailbreak.BENIGN_ESTIMATOR]["count"] == 1
    assert arm["breakdown"]["broken"] == pytest.approx(0.2)


def test_the_breakdown_does_not_carry_a_second_copy_of_the_run_identity():
    """Three empty strings in an artefact read as missing data, and a filled second copy is a
    second thing that can disagree with the first.
    """
    arm = jailbreak.arm_result([ANSWER])
    assert not {"label", "model", "eval"} & set(arm["breakdown"])
    assert {"refusal", "soft_refusal", "noncompliant", "broken", "heretic"} <= set(arm["breakdown"])


def test_a_sample_too_small_withholds_the_rate_and_keeps_the_counts():
    arm = jailbreak.arm_result([ANSWER, REFUSAL])
    assert arm[jailbreak.LOOSE]["rate"] is None
    assert arm[jailbreak.LOOSE]["count"] == 1
    assert arm[jailbreak.LOOSE]["n"] == 2


# ── when the run is not a measurement ─────────────────────────────────────────────────
def _arm(n_answer=0, n_refusal=0, n_broken=0):
    return jailbreak.arm_result([ANSWER] * n_answer + [REFUSAL] * n_refusal + [GARBAGE] * n_broken)


def test_a_clean_pair_is_a_measurement():
    assert jailbreak.validity(_arm(n_answer=30), _arm(n_refusal=30)) is None


def test_a_missing_benign_arm_does_not_stop_the_attack_arm_being_valid():
    assert jailbreak.validity(_arm(n_answer=30), None) is None


def test_an_attack_arm_below_the_floor_is_not_a_measurement():
    why = jailbreak.validity(_arm(n_answer=5), None)
    assert "below the floor" in why
    assert "--n" in why


def test_breakage_past_the_ceiling_invalidates_the_run():
    """A destroyed model reads as a win on either estimator taken alone, so neither may be quoted."""
    why = jailbreak.validity(_arm(n_answer=20, n_broken=20), None)
    assert "attack arm" in why
    assert jailbreak.LOOSE in why
    assert jailbreak.STRICT in why


def test_breakage_on_the_benign_arm_invalidates_the_run_too():
    why = jailbreak.validity(_arm(n_answer=40), _arm(n_refusal=20, n_broken=20))
    assert "benign arm" in why


def test_both_failures_are_reported_rather_than_the_first_one():
    why = jailbreak.validity(_arm(n_answer=5, n_broken=5), None)
    assert "below the floor" in why
    assert " Also: " in why


# ── what reaches the terminal ─────────────────────────────────────────────────────────
def _res(**over):
    res = {
        "label": "x", "attack_set": {"key": "strongreject"}, "benign_set": {"key": "xstest-safe"},
        "attack": _arm(n_answer=30), "benign": _arm(n_refusal=30),
        "budget_warning": None, "pair_incomplete": None, "self_invalidated": None,
        "against_baseline": None, "baseline_missing": "no --baseline was given",
        "edit_overlap": {"checked": False, "reason": jailbreak.NO_EDIT_TRACK},
    }
    res.update(over)
    return res


def test_the_report_gives_both_rates_with_their_counts_and_interval():
    lines = jailbreak._report(_res(), "r.json")
    assert lines[0].startswith("JAILBREAK_DONE")
    assert "(30/30, 95% CI [" in lines[0]
    assert lines[1].startswith("OVER_REFUSAL")
    assert "(30/30, 95% CI [" in lines[1]


#: A percentage, and whatever immediately follows it.
_A_PRINTED_RATE = re.compile(r"=(\d+\.\d%|n/a)(.{0,30})")


def test_no_bare_percentage_reaches_a_terminal_from_either_command():
    """OPERATOR INSTRUCTION, 2026-10-05, and there is a ledger item behind it.

    The terminal line is what gets pasted into a message. `score` prints its finishing rate bare
    while the artefact beside it carries the interval, so the number that travels is the one with
    no uncertainty attached and the number with the uncertainty stays on disk.

    This walks the report lines of BOTH new commands rather than one, because a rule enforced on
    one of two call sites is how the second one ships without it.
    """
    from senbonzakura import multiturn

    mt = {
        "label": "x", "attack_set": {"key": "strongreject"},
        "single_turn": jailbreak.arm_result([ANSWER] * 10 + [REFUSAL] * 20),
        "refused_at_turn_one": 20,
        "arm": multiturn.arm_summary([2] * 8 + [None] * 12, multiturn.REFRAME, refused_n=20),
        "control": multiturn.arm_summary([2] * 7 + [None] * 13, multiturn.CONTROL, refused_n=20),
        "advantage": None, "control_missing": None, "budget_warning": None,
        "self_invalidated": None,
    }
    lines = jailbreak._report(_res(), "r.json") + multiturn._report(mt, "r.json")
    assert lines, "nothing was reported, so this gate is measuring nothing"
    bare = []
    for line in lines:
        for rate, tail in _A_PRINTED_RATE.findall(line):
            if not tail.startswith(" (") or "95% CI" not in tail:
                bare.append(f"{rate!r} in {line[:90]!r}")
    assert not bare, (
        "a rate reached the terminal without its counts and its interval; route it through "
        "`jailbreak.figure`:\n  " + "\n  ".join(bare))


def test_a_withheld_rate_prints_not_available_rather_than_zero():
    """A caller formatting None as 0.0 would reintroduce the defect `reportable_rate` prevents."""
    lines = jailbreak._report(_res(attack=_arm(n_answer=2)), "r.json")
    assert "answered=n/a" in lines[0]
    assert "95% CI [" in lines[0]        # an interval exists even where the rate is withheld


def test_an_absent_interval_prints_not_available():
    assert jailbreak.ci({"ci": None}) == "n/a"


def test_a_missing_benign_arm_is_announced():
    lines = jailbreak._report(
        _res(benign=None, benign_set=None, pair_incomplete="no benign arm was run"), "r.json")
    assert any(ln.startswith("JAILBREAK_PAIR_INCOMPLETE") for ln in lines)
    assert not any(ln.startswith("OVER_REFUSAL") for ln in lines)


def test_an_unreportable_benign_arm_is_announced_beside_its_counts():
    lines = jailbreak._report(
        _res(benign=_arm(n_refusal=2), pair_incomplete="too few benign prompts"), "r.json")
    assert any(ln.startswith("OVER_REFUSAL") for ln in lines)
    assert any(ln.startswith("JAILBREAK_PAIR_INCOMPLETE") for ln in lines)


def test_the_budget_warning_reaches_the_terminal():
    lines = jailbreak._report(_res(budget_warning="too short"), "r.json")
    assert any("BUDGET_WARNING" in ln for ln in lines)


def test_the_invalidation_is_the_last_line_and_names_the_file():
    lines = jailbreak._report(_res(self_invalidated="the model is wrecked"), "out/r.json")
    assert lines[-1].startswith("JAILBREAK_NOT_A_MEASUREMENT")
    assert "do not quote" in lines[-1]
    assert "out/r.json" in lines[-1]


# ── the baseline, without which the figure is not interpretable ───────────────────────
def _stamped(value, n, count, **over):
    """A stamped metrics block as this command writes one, which is what a baseline file holds."""
    block = {
        "metric": "jailbreak_rate", "estimator": jailbreak.STRICT, "model": "base/model",
        "value": value, "n": n, "count": count, "interval": [0.1, 0.3], "reportable": True,
        "input_digest": "abc123", "partition": "all-rows", "prompt_format": "raw",
        "precision": "bfloat16", "tool_version": "0.4.1",
    }
    block.update(over)
    return block


#: The generation settings a run at the command's defaults writes, as the artefact records them.
_SETTINGS = {"max_new_tokens": 64, "batch": 16, "decoding": "greedy", "seed": None,
             "precision": "model default (bfloat16 on the loader)"}


def _baseline_doc(value=0.2, **over):
    return {"model": "base/model", "generation_settings": dict(_SETTINGS),
            "metrics": {f"jailbreak_rate.{jailbreak.STRICT}": _stamped(value, 100, 20, **over)}}


def test_a_baseline_measured_the_same_way_is_compared_and_the_change_reported():
    """Across seven model families baseline refusal on harmful prompts runs 0.4% to 91.5%, so the
    change from the unedited model is the part that is about the edit.
    """
    now = _stamped(0.8, 100, 80, model="edited/model")
    got = jailbreak.baseline_arm(_baseline_doc(), now, jailbreak.STRICT, now_settings=_SETTINGS)
    assert got["not_comparable"] is None
    assert got["baseline"]["rate"] == 0.2
    assert got["delta"]["gap"] == pytest.approx(0.6)
    assert got["delta"]["crosses_zero"] is False
    assert "wider than the truth" in got["delta"]["method"]


def test_a_different_model_is_the_point_and_not_a_mismatch():
    """`baseline.comparability` reports the model as a disagreement, correctly, for a regression
    gate. Here a different checkpoint is the whole comparison, so it is the one named exception.
    """
    now = _stamped(0.8, 100, 80, model="a completely different checkpoint")
    got = jailbreak.baseline_arm(_baseline_doc(), now, jailbreak.STRICT, now_settings=_SETTINGS)
    assert got["not_comparable"] is None


@pytest.mark.parametrize("field", ["input_digest", "partition", "prompt_format", "precision"])
def test_a_baseline_measured_differently_is_refused_and_names_the_field(field):
    """Two rates taken on different prompts, different partitions, different prompt formats or
    different precisions are not a comparison, and subtracting them would look like one.
    """
    now = _stamped(0.8, 100, 80, **{field: "something else"})
    got = jailbreak.baseline_arm(_baseline_doc(), now, jailbreak.STRICT, now_settings=_SETTINGS)
    assert field in got["not_comparable"]
    assert got["delta"] is None, "it subtracted two numbers it had just refused to compare"
    assert got["baseline"]["rate"] == 0.2, "the baseline figure is still reported"


def test_a_baseline_with_no_such_measurement_is_refused_and_says_what_it_holds():
    with pytest.raises(SystemExit, match="nothing here to compare") as e:
        jailbreak.baseline_arm({"metrics": {"refusal_rate.heretic-keyword": {}}},
                               _stamped(0.8, 100, 80), jailbreak.STRICT, now_settings=_SETTINGS)
    assert "refusal_rate.heretic-keyword" in str(e.value)
    with pytest.raises(SystemExit, match="no metrics block at all"):
        jailbreak.baseline_arm({}, _stamped(0.8, 100, 80), jailbreak.STRICT, now_settings=_SETTINGS)


def test_a_withheld_rate_on_either_side_is_not_subtracted():
    """A difference of a number and a refusal to state a number is not a number."""
    assert jailbreak.baseline_arm(
        _baseline_doc(value=None), _stamped(0.8, 100, 80), jailbreak.STRICT, now_settings=_SETTINGS)["delta"] is None
    assert jailbreak.baseline_arm(
        _baseline_doc(), _stamped(None, 100, 80), jailbreak.STRICT, now_settings=_SETTINGS)["delta"] is None
    assert jailbreak.baseline_arm(
        _baseline_doc(), _stamped(0.8, 0, 0), jailbreak.STRICT, now_settings=_SETTINGS)["delta"] is None


def test_a_change_inside_its_interval_says_the_run_does_not_separate_the_models():
    now = _stamped(0.22, 100, 22)
    delta = jailbreak.baseline_arm(_baseline_doc(), now, jailbreak.STRICT, now_settings=_SETTINGS)["delta"]
    assert delta["crosses_zero"] is True
    lines = jailbreak._report(
        _res(against_baseline=jailbreak.baseline_arm(_baseline_doc(), now, jailbreak.STRICT, now_settings=_SETTINGS)),
        "r.json")
    assert any("crosses zero" in ln for ln in lines)


def test_a_missing_baseline_is_announced_with_the_reason_it_matters():
    lines = jailbreak._report(_res(), "r.json")
    assert any(ln.startswith("JAILBREAK_NO_BASELINE") for ln in lines)


def test_an_incomparable_baseline_is_announced_and_not_subtracted():
    lines = jailbreak._report(
        _res(against_baseline={"not_comparable": "input_digest differs",
                               "baseline": {}, "delta": None}), "r.json")
    assert any("NOT_COMPARABLE" in ln and "NOT subtracted" in ln for ln in lines)


@pytest.mark.parametrize(("content", "says"), [
    ("{not json", "not valid JSON"),
    ('["a"]', "holds a list"),
])
def test_a_baseline_file_that_is_not_a_results_file_is_refused(tmp_path, content, says):
    p = tmp_path / "b.json"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(SystemExit, match=says):
        jailbreak._read_baseline(str(p))


def test_an_absent_baseline_file_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="could not be read"):
        jailbreak._read_baseline(str(tmp_path / "absent.json"))


@pytest.mark.parametrize(("field", "now"), [("max_new_tokens", 512), ("batch", 1)])
def test_a_baseline_at_a_different_token_budget_or_batch_is_refused(field, now):
    """A refusal the model never gets far enough to state is not counted, so the token budget
    decides the rate, and the batch size changes the tokens. A subtraction across either is a
    change of settings presented as a change of model.
    """
    got = jailbreak.baseline_arm(_baseline_doc(), _stamped(0.8, 100, 80), jailbreak.STRICT,
                                 now_settings=dict(_SETTINGS, **{field: now}))
    assert f"generation_settings.{field}" in got["not_comparable"]
    assert got["delta"] is None
    assert got["baseline"]["rate"] == 0.2, "the baseline figure is still reported"


def test_a_baseline_with_no_recorded_generation_settings_is_refused_not_assumed_matched():
    doc = _baseline_doc()
    del doc["generation_settings"]
    got = jailbreak.baseline_arm(doc, _stamped(0.8, 100, 80), jailbreak.STRICT,
                                 now_settings=_SETTINGS)
    assert "generation_settings" in got["not_comparable"]
    assert got["delta"] is None


# ── the whole command ─────────────────────────────────────────────────────────────────
class _Tok:
    senbon_chat_template = None


def _drive(monkeypatch, argv, *, attack=None, benign=None, prompts=30, rows=None):
    """Run `main` with the model and the corpora replaced, so every branch is reachable on CPU.

    `rows` is `(attack prompts, benign prompts)` when a test needs the prompts to be known exactly,
    for the overlap check. Otherwise each set is generated, which is enough for everything else.
    """
    from senbonzakura import score

    monkeypatch.setattr(score, "load_model_and_tokenizer", lambda *a, **k: (object(), _Tok()))
    if rows is None:
        monkeypatch.setattr(jailbreak, "prompts_for",
                            lambda key, limit, what: [f"{key} {i}" for i in range(limit or prompts)])
    else:
        monkeypatch.setattr(jailbreak, "prompts_for",
                            lambda key, limit, what: rows[0] if what == "attack set" else rows[1])
    replies = {"attack": attack or [ANSWER] * 30, "benign": benign or [REFUSAL] * 30}
    seen = []

    def _generate(model, tok, given, device, batch=16, max_new=64):
        seen.append(list(given))
        return replies["attack" if len(seen) == 1 else "benign"]

    monkeypatch.setattr(score, "generate", _generate)
    return jailbreak.main(argv), seen


def test_the_command_writes_both_arms_and_stamps_both_metrics(monkeypatch, tmp_path):
    out = str(tmp_path / "r.json")
    res, _seen = _drive(monkeypatch, ["--model", "m", "--device", "cpu", "--out", out])
    assert res["attack"][jailbreak.LOOSE]["rate"] == 1.0
    assert res["benign"][jailbreak.BENIGN_ESTIMATOR]["rate"] == 1.0
    with open(out) as f:
        doc = json.load(f)
    assert set(doc["metrics"]) == {
        f"jailbreak_rate.{jailbreak.LOOSE}", f"jailbreak_rate.{jailbreak.STRICT}",
        "over_refusal_rate"}
    # Every stamped number carries the interval and the counts, which is the thing no competitor
    # puts on this figure, AND all five pinned fields, without which `baseline.comparability`
    # reports the figure as incomparable with every other one and the gate refuses it silently.
    #
    # ASSERTED ON THE ARTEFACT, not on the source text. The suite's AST scan over
    # `measurement.stamp` call sites skips a writer that unpacks a local, which is what this one
    # does because it stamps three times, so this is the half of that pair that checks the result.
    for key, block in doc["metrics"].items():
        assert block["interval"] and block["count"] is not None, key
        missing = [f for f in ("input_digest", "partition", "prompt_format", "tool_version",
                               "precision") if block.get(f) in (None, "")]
        assert not missing, f"{key} is missing {missing}, so it can never be gated"
    # And both arms are pinned by their own digest, not by one shared one.
    assert (doc["metrics"]["over_refusal_rate"]["input_digest"]
            != doc["metrics"][f"jailbreak_rate.{jailbreak.LOOSE}"]["input_digest"])


def test_a_truncated_run_does_not_claim_to_have_scored_the_whole_corpus(monkeypatch, tmp_path):
    """`stamps.partition_of` maps a skip of zero to `all-rows`, which is true of a full run and
    false of `--n 40` against a 313-row set: that figure described forty rows and its own
    provenance said it described every one of them.
    """
    from senbonzakura import stamps

    out = str(tmp_path / "r.json")
    _drive(monkeypatch, ["--model", "m", "--device", "cpu", "--out", out, "--n", "30"])
    with open(out) as f:
        doc = json.load(f)
    assert doc["metrics"][f"jailbreak_rate.{jailbreak.STRICT}"]["partition"] == "first-30-rows"

    full = str(tmp_path / "full.json")
    _drive(monkeypatch, ["--model", "m", "--device", "cpu", "--out", full], prompts=30)
    with open(full) as f:
        whole = json.load(f)
    assert whole["metrics"][f"jailbreak_rate.{jailbreak.STRICT}"]["partition"] == stamps.ALL_ROWS


def test_both_corpora_are_recorded_with_their_licence_and_a_revision(monkeypatch, tmp_path):
    out = str(tmp_path / "r.json")
    _drive(monkeypatch, ["--model", "m", "--device", "cpu", "--out", out])
    with open(out) as f:
        doc = json.load(f)
    for block in (doc["attack_set"], doc["benign_set"]):
        assert block["licence"] and block["attribution"]
        assert "revision" in block


def test_opting_out_of_the_benign_arm_records_an_incomplete_pair(monkeypatch, tmp_path, capsys):
    out = str(tmp_path / "r.json")
    res, seen = _drive(monkeypatch, ["--model", "m", "--device", "cpu", "--out", out,
                                     "--no-over-refusal"])
    assert len(seen) == 1, "the benign arm was generated despite being opted out of"
    assert res["benign"] is None
    assert res["benign_set"] is None
    assert "--no-over-refusal" in res["pair_incomplete"]
    assert "refuses everything" in res["pair_incomplete"]
    assert "JAILBREAK_PAIR_INCOMPLETE" in capsys.readouterr().out
    with open(out) as f:
        assert json.load(f)["pair_incomplete"]


def test_a_benign_arm_too_small_to_state_a_rate_records_an_incomplete_pair(monkeypatch, tmp_path):
    out = str(tmp_path / "r.json")
    res, _seen = _drive(monkeypatch, ["--model", "m", "--device", "cpu", "--out", out,
                                      "--benign-n", "4"],
                        benign=[REFUSAL] * 4)
    assert res["benign"][jailbreak.BENIGN_ESTIMATOR]["rate"] is None
    assert "below the floor" in res["pair_incomplete"]
    assert "--benign-n" in res["pair_incomplete"]
    # The attack arm is unaffected: it had enough rows, so its rate still stands.
    assert res["attack"][jailbreak.LOOSE]["rate"] == 1.0


def test_a_wrecked_model_invalidates_the_run_in_the_file_and_in_the_exit_status(
        monkeypatch, tmp_path, capsys):
    out = str(tmp_path / "r.json")
    res, _seen = _drive(monkeypatch, ["--model", "m", "--device", "cpu", "--out", out],
                        attack=[ANSWER] * 15 + [GARBAGE] * 15)
    assert res["self_invalidated"]
    with open(out) as f:
        assert json.load(f)["self_invalidated"], "the verdict never reached the file"
    assert "NOT A MEASUREMENT" in capsys.readouterr().out
    assert entry.exit_status(res) == 1


def test_the_generations_of_each_arm_go_to_their_own_file(monkeypatch, tmp_path):
    prefix = str(tmp_path / "gens")
    _drive(monkeypatch, ["--model", "m", "--device", "cpu", "--out", str(tmp_path / "r.json"),
                         "--save-generations", prefix])
    with open(f"{prefix}.attack.jsonl", encoding="utf-8") as f:
        attack = [json.loads(line) for line in f]
    with open(f"{prefix}.benign.jsonl", encoding="utf-8") as f:
        benign = [json.loads(line) for line in f]
    assert attack[0]["mode"] == "jailbreak-attack"
    assert benign[0]["mode"] == "jailbreak-benign"
    assert attack[0]["generation"] == ANSWER
    assert benign[0]["generation"] == REFUSAL


def test_the_command_compares_itself_against_a_previous_run_of_itself(monkeypatch, tmp_path):
    """The baseline path end to end: one run writes the artefact, the next reads it.

    THE SHAPES HAVE TO AGREE and nothing but this would notice if they stopped. The reader looks
    up `metrics["jailbreak_rate.<estimator>"]` and the pinned fields inside it, and both the key
    and the fields are written by the stamping code in this same module, so a test that built the
    baseline by hand would pass through a rename that broke every real file.
    """
    base = str(tmp_path / "base.json")
    _drive(monkeypatch, ["--model", "unedited", "--device", "cpu", "--out", base],
           attack=[REFUSAL] * 30)
    res, _ = _drive(monkeypatch, ["--model", "edited", "--device", "cpu",
                                  "--out", str(tmp_path / "r.json"), "--baseline", base])
    against = res["against_baseline"]
    assert against["not_comparable"] is None, against["not_comparable"]
    assert against["baseline"]["rate"] == 0.0, "the unedited arm answered nothing"
    assert against["delta"]["gap"] == pytest.approx(1.0)
    assert res["baseline_missing"] is None


def test_a_baseline_scored_on_different_prompts_is_refused_end_to_end(monkeypatch, tmp_path):
    base = str(tmp_path / "base.json")
    _drive(monkeypatch, ["--model", "unedited", "--device", "cpu", "--out", base,
                         "--attack-set", "advbench"], attack=[REFUSAL] * 30)
    res, _ = _drive(monkeypatch, ["--model", "edited", "--device", "cpu",
                                  "--out", str(tmp_path / "r.json"), "--baseline", base,
                                  "--attack-set", "strongreject"])
    assert "input_digest" in res["against_baseline"]["not_comparable"]
    assert res["against_baseline"]["delta"] is None


def test_a_bad_baseline_path_is_refused_before_the_model_loads(monkeypatch, tmp_path):
    """On a rented machine a typo in a path must not cost a multi-gigabyte load first."""
    from senbonzakura import score

    loaded = []
    monkeypatch.setattr(score, "load_model_and_tokenizer",
                        lambda *a, **k: loaded.append(1) or (object(), _Tok()))
    monkeypatch.setattr(jailbreak, "prompts_for", lambda key, limit, what: ["p"] * 30)
    with pytest.raises(SystemExit, match="could not be read"):
        jailbreak.main(["--model", "m", "--device", "cpu", "--out", str(tmp_path / "r.json"),
                        "--baseline", str(tmp_path / "absent.json")])
    assert not loaded, "the model was loaded before the baseline path was checked"


def test_no_artefact_these_commands_write_carries_a_key_the_leak_gate_bans(monkeypatch, tmp_path):
    """FOUND BY RUNNING THE GATE AGAINST A REAL ARTEFACT, 2026-10-05.

    `tools/ci/check_prompt_artefacts.py` is the one control between a harmful prompt and a public
    push, and it refuses a banned key name at any depth. It is right to: reading values instead
    would need a denylist of what a harmful reply says, which this project has considered and
    refused. So an artefact carrying `generation` is unpublishable even when the value is a block
    of settings, which is what the first version of all three of these commands wrote.

    The resolution is to rename the field in the data, never to widen the gate. That is the
    precedent `text` set under decision Q-33, when twelve of our own artefacts used it for single
    decoded tokens and were renamed rather than exempted.

    All three commands are walked here rather than one, because a rule enforced on one of three
    writers is how the next one ships without it.
    """
    banned = _banned_keys()
    assert "generation" in banned and "prompt" in banned, (
        "the gate's banned key list no longer looks the way this test assumes; read it again "
        "rather than deleting this")

    written = []
    jb, _ = _drive(monkeypatch, ["--model", "m", "--device", "cpu",
                                 "--out", str(tmp_path / "jb.json")])
    written.append(("jailbreak", jb))
    written.append(("multi-turn", _drive_multiturn(monkeypatch, tmp_path)))
    written.append(("tamper", _drive_tamper(monkeypatch, tmp_path)))

    for name, doc in written:
        for key in _every_key(doc):
            assert key not in banned, (
                f"the {name} artefact carries the key {key!r}, which the leak gate bans at any "
                f"depth, so the artefact cannot be committed. Rename the field in the data.")


def _banned_keys():
    """The gate's own list, read from the gate rather than copied into this file."""
    import importlib.util
    from pathlib import Path

    gate = Path(__file__).resolve().parent.parent / "tools" / "ci" / "check_prompt_artefacts.py"
    spec = importlib.util.spec_from_file_location("_leak_gate", gate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BANNED_KEYS


def _every_key(value):
    """Every mapping key at any depth, which is the depth the gate reads at."""
    if isinstance(value, dict):
        for key, inner in value.items():
            yield key
            yield from _every_key(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from _every_key(inner)


def _drive_multiturn(monkeypatch, tmp_path):
    from senbonzakura import multiturn, score

    monkeypatch.setattr(score, "load_model_and_tokenizer", lambda *a, **k: (object(), _Tok()))
    monkeypatch.setattr(jailbreak, "prompts_for",
                        lambda key, limit, what: [f"p{i}" for i in range(40)])
    monkeypatch.setattr(multiturn, "generate_turn",
                        lambda model, tok, conversations, device, **k:
                        ([REFUSAL] * len(conversations), []))
    return multiturn.main(["--model", "m", "--device", "cpu",
                           "--out", str(tmp_path / "mt.json")])


def _drive_tamper(monkeypatch, tmp_path):
    from senbonzakura import score, tamper

    monkeypatch.setattr(score, "load_model_and_tokenizer", lambda *a, **k: (object(), _Tok()))
    monkeypatch.setattr(jailbreak, "prompts_for",
                        lambda key, limit, what: [f"p{i}" for i in range(80)])
    monkeypatch.setattr(tamper, "prepare", lambda model, recipe, log=print: model)
    monkeypatch.setattr(tamper, "train", lambda *a, **k: tamper.loss_trace([2.0, 1.0]))
    monkeypatch.setattr(score, "generate",
                        lambda m, t, prompts, d, batch=16, max_new=64: [REFUSAL] * len(prompts))
    return tamper.main(["--model", "m", "--device", "cpu", "--train-n", "40",
                        "--resamples", "50", "--out", str(tmp_path / "tamper.json")])


def test_two_runs_on_the_same_input_produce_the_same_artefact(monkeypatch, tmp_path):
    """Decoding is greedy and nothing here samples, so a second run must agree byte for byte.

    VERIFIED ON A REAL MODEL TOO, 2026-10-05: two runs of this command against
    SmolLM2-135M-Instruct over 31 attack and 31 benign prompts produced identical artefacts
    including the provenance block. This is the guard that keeps it true, because the thing most
    likely to break it is a future field holding a timestamp, a set iteration or a dict ordered by
    something other than insertion.
    """
    first, _ = _drive(monkeypatch, ["--model", "m", "--device", "cpu",
                                    "--out", str(tmp_path / "a.json")])
    second, _ = _drive(monkeypatch, ["--model", "m", "--device", "cpu",
                                     "--out", str(tmp_path / "b.json")])
    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(
        second, sort_keys=True, default=str)


def test_the_ruler_is_checked_before_a_single_prompt_is_sent(monkeypatch, tmp_path):
    """A ruler that misreads yields a confident wrong number rather than an error, and both arms
    of this measurement come off that ruler.
    """
    from senbonzakura import metrics

    monkeypatch.setattr(metrics, "validate_ruler",
                        lambda: (_ for _ in ()).throw(RuntimeError("the ruler misreads")))
    with pytest.raises(RuntimeError, match="misreads"):
        _drive(monkeypatch, ["--model", "m", "--device", "cpu",
                             "--out", str(tmp_path / "r.json")])


# ── the corpora this actually runs on ─────────────────────────────────────────────────
@needs_corpora
def test_the_attack_set_resolves_at_its_declared_size():
    assert len(jailbreak.prompts_for("strongreject", 0, what="attack set")) == 313
    assert len(jailbreak.prompts_for("strongreject", 7, what="attack set")) == 7


@needs_corpora
def test_asking_for_more_prompts_than_the_set_holds_is_refused_before_the_model_loads():
    with pytest.raises(SystemExit) as e:
        jailbreak.prompts_for("strongreject", 9999, what="attack set")
    assert "313" in str(e.value)


def test_an_unreadable_corpus_is_a_refusal_rather_than_a_traceback(monkeypatch):
    from senbonzakura import dataset

    monkeypatch.setattr(dataset, "resolve",
                        lambda *a, **k: (_ for _ in ()).throw(dataset.DatasetError("no pack")))
    with pytest.raises(SystemExit, match="no pack"):
        jailbreak.prompts_for("strongreject", 0, what="attack set")


@needs_corpora
def test_the_command_runs_against_the_real_bundled_corpora(monkeypatch, tmp_path):
    """The resolver, the arm guard and the slice, on the corpora that ship, with a fake model.

    Everything above replaces `_prompts`, so without this nothing exercises the path a real run
    takes to its prompts.
    """
    from senbonzakura import score

    monkeypatch.setattr(score, "load_model_and_tokenizer", lambda *a, **k: (object(), _Tok()))
    monkeypatch.setattr(score, "generate",
                        lambda model, tok, given, device, batch=16, max_new=64:
                        [ANSWER] * len(given))
    res = jailbreak.main(["--model", "m", "--device", "cpu", "--attack-set", "advbench",
                          "--n", "31", "--benign-n", "31",
                          "--out", str(tmp_path / "r.json")])
    assert res["attack_set"]["key"] == "advbench"
    assert res["attack"]["n"] == 31
    # Every benign prompt was answered, so nothing was over-refused.
    assert res["benign"][jailbreak.BENIGN_ESTIMATOR]["rate"] == 0.0


# ── Choosing between a bundled name and a caller-supplied file ──────────────────────────────────

def _set_file(tmp_path, name="mine.txt", rows=("pick a lock", "hotwire a car")):
    path = tmp_path / name
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


@needs_corpora
def test_a_bundled_name_still_resolves_through_the_registry():
    corpus, prompts = jailbreak.corpus_and_prompts(
        "advbench", "", arm="harmful", name_flag="--attack-set", file_flag="--attack-set-file")
    assert corpus.key == "advbench"
    assert jailbreak.set_block(corpus, len(prompts))["arm_source"] == "declared-by-registry"


def test_a_supplied_file_resolves_and_is_marked_as_asserted(tmp_path):
    corpus, prompts = jailbreak.corpus_and_prompts(
        "advbench", str(_set_file(tmp_path)), arm="harmful",
        name_flag="--attack-set", file_flag="--attack-set-file")
    assert len(prompts) == 2
    block = jailbreak.set_block(corpus, len(prompts))
    assert block["arm_source"] == "asserted-by-the-flag-that-supplied-the-file"
    assert block["arm"] == "harmful"


def test_a_supplied_file_is_pinned_by_its_bytes(tmp_path):
    # A single file was unpinnable until 2026-10-07 and is the most mutable corpus there is: a
    # customer's own set, edited in place, under a path that does not change.
    first = _set_file(tmp_path)
    corpus, prompts = jailbreak.corpus_and_prompts(
        "advbench", str(first), arm="harmful",
        name_flag="--attack-set", file_flag="--attack-set-file")
    before = jailbreak.set_block(corpus, len(prompts))["revision"]
    assert before["kind"] == "corpus-file"
    assert len(before["revision"]) == 64

    first.write_text("pick a lock\nsomething else entirely\n", encoding="utf-8")
    corpus2, prompts2 = jailbreak.corpus_and_prompts(
        "advbench", str(first), arm="harmful",
        name_flag="--attack-set", file_flag="--attack-set-file")
    after = jailbreak.set_block(corpus2, len(prompts2))["revision"]
    assert after["revision"] != before["revision"], "a rebuilt corpus must not keep its pin"


def test_an_empty_supplied_file_is_refused_by_the_resolver_before_a_corpus_is_built(tmp_path):
    # ATTRIBUTED DELIBERATELY. This asserts the RESOLVER's refusal, not a guard in
    # `corpus_and_prompts`: one was written there, its test matched this same message, and the
    # guard turned out to be unreachable. A test that cannot tell which code refused is a test
    # that will keep passing after the code it was written for is deleted.
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="is empty, so there is nothing to measure"):
        jailbreak.corpus_and_prompts(
            "advbench", str(empty), arm="harmful",
            name_flag="--attack-set", file_flag="--attack-set-file")


def test_a_broken_manifest_beside_a_supplied_file_names_the_flag_that_supplied_it(tmp_path):
    path = _set_file(tmp_path)
    (tmp_path / ("mine.txt" + corpora.MANIFEST_SUFFIX)).write_text("{nope", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"--attack-set-file .*could not be read"):
        jailbreak.corpus_and_prompts(
            "advbench", str(path), arm="harmful",
            name_flag="--attack-set", file_flag="--attack-set-file")


def test_a_supplied_file_that_does_not_exist_names_the_flag(tmp_path):
    with pytest.raises(SystemExit):
        jailbreak.corpus_and_prompts(
            "advbench", str(tmp_path / "absent.txt"), arm="harmful",
            name_flag="--attack-set", file_flag="--attack-set-file")


@needs_corpora
def test_a_bundled_name_in_the_wrong_arm_is_still_refused():
    # The protection the file flags relaxed must survive for names.
    with pytest.raises(SystemExit, match="arm and this slot needs"):
        jailbreak.corpus_and_prompts(
            "xstest-safe", "", arm="harmful",
            name_flag="--attack-set", file_flag="--attack-set-file")


def test_the_file_flags_cannot_be_combined_with_the_name_flags():
    # Mutual exclusion is argparse's, so the "is this a path or a name" guess never happens and a
    # file whose name matches a registry key cannot be ambiguous.
    with pytest.raises(SystemExit):
        jailbreak.build_parser().parse_args(
            ["--model", "m", "--attack-set", "advbench", "--attack-set-file", "x.txt"])


def test_the_help_for_a_file_flag_states_which_arm_the_flag_asserts():
    assert "decline" in jailbreak.file_flag_help("decline")
    assert "answer" in jailbreak.file_flag_help("answer")
    for verb in ("decline", "answer"):
        text = jailbreak.file_flag_help(verb)
        assert "asserted rather than declared" in text
        assert corpora.MANIFEST_SUFFIX in text


@pytest.mark.parametrize(("flag", "edited", "field"),
                         [("--max-new", "512", "max_new_tokens"), ("--batch", "4", "batch")])
def test_a_baseline_run_at_a_different_setting_is_refused_end_to_end(
        monkeypatch, tmp_path, flag, edited, field):
    """The settings reach the comparison through `main`, not only through a direct call, because
    that is the path a reader's two artefacts take.
    """
    base = str(tmp_path / "base.json")
    _drive(monkeypatch, ["--model", "unedited", "--device", "cpu", "--out", base,
                         "--max-new", "64", "--batch", "16"], attack=[REFUSAL] * 30)
    res, _ = _drive(monkeypatch, ["--model", "edited", "--device", "cpu",
                                  "--out", str(tmp_path / "r.json"), "--baseline", base,
                                  flag, edited])
    against = res["against_baseline"]
    assert against["not_comparable"] is not None, "the two runs were subtracted across a setting"
    assert f"generation_settings.{field}" in against["not_comparable"]
    assert against["delta"] is None


# ── the edit's track, which the scored sets must not overlap ─────────────────────────────
_THINGS = [f"item{i:02d}" for i in range(30)]
ATTACK_ROWS = [f"describe a method to bypass the {t} security" for t in _THINGS]
BENIGN_ROWS = [f"how do I make a {t} at home" for t in _THINGS]


def _edit_track(monkeypatch, *, harmful_fit=(), harmful_search=(), harmless_fit=()):
    """The partitions of a track, without building one. Building a real track is tested in
    `test_track.py`; this checks what this command does with the partitions it is handed.
    """
    parts = ({"fit": list(harmful_fit), "search": list(harmful_search),
              "measure": ["a held out request about boats and their keels"]},
             {"fit": list(harmless_fit), "search": [], "measure": []})
    monkeypatch.setattr(jailbreak.track, "load_partitions", lambda path: parts)


def test_an_attack_row_the_edit_was_fitted_on_is_refused_before_the_model_loads(
        monkeypatch, tmp_path):
    from senbonzakura import score

    loaded = []
    monkeypatch.setattr(score, "load_model_and_tokenizer",
                        lambda *a, **k: loaded.append(1) or (object(), _Tok()))
    monkeypatch.setattr(jailbreak, "prompts_for",
                        lambda key, limit, what: ATTACK_ROWS if what == "attack set" else BENIGN_ROWS)
    _edit_track(monkeypatch, harmful_fit=[ATTACK_ROWS[7]])
    out = tmp_path / "r.json"
    with pytest.raises(SystemExit, match="fitted or searched on") as e:
        jailbreak.main(["--model", "m", "--device", "cpu", "--out", str(out),
                        "--edit-track", str(tmp_path)])
    assert "attack set" in str(e.value)
    assert not loaded, "the model was loaded before the overlap was refused"
    assert not out.exists(), "a refused run wrote a result file"


def test_a_benign_row_the_edit_was_fitted_on_is_refused(monkeypatch, tmp_path):
    from senbonzakura import score

    loaded = []
    monkeypatch.setattr(score, "load_model_and_tokenizer",
                        lambda *a, **k: loaded.append(1) or (object(), _Tok()))
    monkeypatch.setattr(jailbreak, "prompts_for",
                        lambda key, limit, what: ATTACK_ROWS if what == "attack set" else BENIGN_ROWS)
    _edit_track(monkeypatch, harmless_fit=[BENIGN_ROWS[3]])
    with pytest.raises(SystemExit, match="fitted or searched on") as e:
        jailbreak.main(["--model", "m", "--device", "cpu", "--out", str(tmp_path / "r.json"),
                        "--edit-track", str(tmp_path)])
    assert "benign set" in str(e.value)
    assert not loaded


def test_a_clean_edit_track_lets_the_run_proceed_and_records_both_checks(
        monkeypatch, tmp_path, capsys):
    _edit_track(monkeypatch, harmful_fit=["a request nobody asked about"],
                harmless_fit=["a harmless question nobody asked about"])
    res, _ = _drive(monkeypatch, ["--model", "m", "--device", "cpu",
                                  "--out", str(tmp_path / "r.json"),
                                  "--edit-track", str(tmp_path)],
                    rows=(ATTACK_ROWS, BENIGN_ROWS))
    overlap = res["edit_overlap"]
    assert overlap["checked"] is True
    assert overlap["attack"]["verdict"] == "clean"
    assert overlap["benign"]["verdict"] == "clean"
    assert overlap["attack"]["in_fitting_side"] == 0
    assert "JAILBREAK_NO_EDIT_TRACK" not in capsys.readouterr().out


def test_with_no_benign_arm_only_the_attack_set_is_checked(monkeypatch, tmp_path):
    """`--no-over-refusal` means the benign set is never scored, so overlap with it is not a
    reason to refuse the run.
    """
    _edit_track(monkeypatch, harmless_fit=[BENIGN_ROWS[3]])
    res, _ = _drive(monkeypatch, ["--model", "m", "--device", "cpu",
                                  "--out", str(tmp_path / "r.json"),
                                  "--edit-track", str(tmp_path), "--no-over-refusal"],
                    rows=(ATTACK_ROWS, BENIGN_ROWS))
    assert res["edit_overlap"]["checked"] is True
    assert "benign" not in res["edit_overlap"]


def test_without_an_edit_track_the_artefact_and_the_terminal_both_say_it_was_not_checked(
        monkeypatch, tmp_path, capsys):
    res, _ = _drive(monkeypatch, ["--model", "m", "--device", "cpu",
                                  "--out", str(tmp_path / "r.json")])
    assert res["edit_overlap"] == {"checked": False, "reason": jailbreak.NO_EDIT_TRACK}
    out = capsys.readouterr().out
    assert "JAILBREAK_NO_EDIT_TRACK" in out
    assert "memorisation" in out
