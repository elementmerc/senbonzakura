"""A drift figure must not claim more precision than the arithmetic that produced it.

A KL is a sum over the vocabulary of quantities that shrink as the two models converge, so below
some magnitude the dtype dominates the measurement. The floor is measured rather than guessed: the
same edit scored in float32 and bfloat16 diverges by 0.0% at KL 0.01, by 0.2% at 0.0029, and by
0.7% at 0.00074, growing as the figure gets smaller.

This became a live question when a second implementation of the same measurement turned up
(`kl_llama.py`) using float32 where `drift` uses bfloat16. Its formula was identical, so the two
agree wherever the number is big enough to matter. That agreement is only knowable because someone
checked, and the floor is what keeps it true for the next figure rather than only for this one.
"""
import pytest

from senbonzakura.drift import BF16_KL_FLOOR, precision_verdict


# ── the published range is unaffected ──────────────────────────────────────────────
@pytest.mark.parametrize("value", [0.0614, 0.065, 0.0892, 0.34, 1.0])
def test_the_figures_this_project_publishes_are_fine_in_bf16(value):
    ok, note = precision_verdict(value, "bfloat16")
    assert ok is True
    assert note is None


def test_exactly_at_the_floor_is_accepted():
    # Inclusive, and stated rather than left to a comparison operator: the measured divergence at
    # 1e-3 is already under a percent, so refusing it would withhold a usable number.
    ok, _ = precision_verdict(BF16_KL_FLOOR, "bfloat16")
    assert ok is True


# ── below the floor, in reduced precision ─────────────────────────────────────────
@pytest.mark.parametrize("value", [0.0009, 5e-4, 1e-5, 2e-6, 0.0])
def test_a_small_figure_from_bf16_is_flagged(value):
    ok, note = precision_verdict(value, "bfloat16")
    assert ok is False
    assert note


def test_float16_is_flagged_on_the_same_grounds():
    ok, _ = precision_verdict(5e-4, "float16")
    assert ok is False


@pytest.mark.parametrize("name", ["bfloat16", "torch.bfloat16", "BF16", "bf16", "fp16", "Float16"])
def test_every_spelling_of_a_reduced_dtype_is_recognised(name):
    """The dtype arrives as a string from `str(model.dtype)` and spellings vary by torch version."""
    assert precision_verdict(5e-4, name)[0] is False


# ── full precision is not limited by this ─────────────────────────────────────────
@pytest.mark.parametrize("name", ["float32", "torch.float32", "fp32", "float64"])
def test_a_small_figure_from_full_precision_is_accepted(name):
    ok, note = precision_verdict(1e-6, name)
    assert ok is True
    assert note is None


# ── the message has to be actionable, since it travels with the number ────────────
def test_the_note_says_what_to_do_instead():
    _, note = precision_verdict(5e-4, "bfloat16")
    assert "float32" in note                 # the route to a real figure
    assert "rather than as a value" in note   # how to read the one in hand


def test_the_note_carries_the_evidence_not_just_the_verdict():
    """A caveat a reader cannot check is a caveat they will ignore."""
    _, note = precision_verdict(5e-4, "bfloat16")
    assert "0.7%" in note
    assert "float32 and bfloat16" in note


def test_the_note_names_the_actual_value_and_floor():
    _, note = precision_verdict(5e-4, "bfloat16")
    assert "5.00e-04" in note
    assert "1e-03" in note


# ── the floor itself ──────────────────────────────────────────────────────────────
def test_the_floor_is_where_the_measurement_put_it():
    # Not a round number chosen for looking tidy: 1e-3 is where the measured bf16 divergence
    # crosses roughly half a percent. Pinned so a later edit has to argue with the measurement.
    assert BF16_KL_FLOOR == 1e-3


def test_the_floor_can_be_overridden_for_a_stricter_caller():
    ok, _ = precision_verdict(0.005, "bfloat16", floor=1e-2)
    assert ok is False


# ── reading the dtype off a model, and admitting when it cannot be read ───────────
def test_an_unreadable_dtype_is_unknown_rather_than_assumed():
    """Flagging on ignorance would fire on every stub while telling nobody anything.

    A model shape this cannot read is not evidence of reduced precision, so it records "unknown"
    and the verdict passes. The artefact then says the check did not run, which is the same
    discipline the promotion stamp uses for the strata check.
    """
    from senbonzakura.drift import logits_dtype_of

    class _NoDtype:
        pass

    assert logits_dtype_of(_NoDtype()) == "unknown"
    assert precision_verdict(1e-9, "unknown")[0] is True


def test_a_models_own_dtype_attribute_wins():
    from senbonzakura.drift import logits_dtype_of

    class _M:
        dtype = "torch.bfloat16"

    assert logits_dtype_of(_M()) == "bfloat16"


def test_the_dtype_falls_back_to_the_first_parameter():
    import torch
    from torch import nn

    from senbonzakura.drift import logits_dtype_of
    m = nn.Linear(2, 2).to(torch.float32)
    assert logits_dtype_of(m) == "float32"
