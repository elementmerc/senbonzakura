# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The residual leak metric, from the flag to the artefact.

WHY THIS FILE EXISTS

`residualleak.py` was reachable from nothing. No `main`, no parser, no `__main__` block, and no
module in the package importing it, while `private/decisions.md` recorded the leak metric as the
instrument this project adopted over embedding ablation. So the decision log and the shipped tool
disagreed, and the metric could not be produced by any user at all. Found by
`tools/research/audit_unreached.py`, which exists because the same shape had just turned up twice
in two days.

The tests here are about the JOIN rather than about the metric. `measure_leak` has its own tests in
`test_a_removed_direction_is_a_claim_about_a_basis.py` and they were passing the whole time it was
unreachable, which is exactly the point: a measure can be thoroughly covered and reach nobody, and
coverage reports the first fact.

WHAT THE REFUSAL IS FOR, since it is the least obvious part

The quantity is the component of the residual along *the* refusal direction, so it is defined
against one vector. A recipe that applied two directions per layer spans a plane, and the leak out
of a plane is a different quantity with a different derivation. Reporting one of the K directions
under the plain name would be a true number about something nobody asked for, printed beside a
claim it does not support, so the run says which recipe it used and why that recipe has no such
figure.
"""
from types import SimpleNamespace

import pytest

from senbonzakura import cli, residualleak


class _Output:
    def __init__(self, post=3e-8, refused=None):
        self.along_post_norm_direction = post
        self.along_pre_norm_direction = 0.0096
        self.refused_because = refused
        self.norm_condition_number = 4.2
        self.norm_diagonal_agreement = 0.999
        self.diagonal_from = "final_norm.weight"


class _Report:
    """Enough of a `LeakReport` for the block builder, which is all these tests drive."""

    #: Sentinel, because a mutable default evaluated once is a shared object across
    #: every test in the file, and `None` already means "no final norm at all" here.
    DEFAULT_OUTPUT = object()

    def __init__(self, output=DEFAULT_OUTPUT, warnings=()):
        self.positions = 4
        self.leak_per_position = (1e-8, 2e-8, 1.5e-8, 1e-8)
        self.probe_prompts = 32
        self.pinned = {"input_digest": "d" * 16, "partition": "measure",
                       "prompt_format": "chat", "precision": "float32",
                       "tool_version": "0.4.1"}
        self.output = _Output() if output is self.DEFAULT_OUTPUT else output
        self.warnings = tuple(warnings)
        self.basis = "the pre-norm residual stream"

    @property
    def mean(self):
        return sum(self.leak_per_position) / len(self.leak_per_position)


# ── which recipes have a figure at all ───────────────────────────────────────────────

def test_one_direction_shared_across_layers_is_measurable():
    assert residualleak.unmeasurable_because("single", 1) is None


@pytest.mark.parametrize(("mode", "k"), [("single", 2), ("single", 8), ("per_layer", 3)])
def test_more_than_one_direction_has_no_single_figure(mode, k):
    why = residualleak.unmeasurable_because(mode, k)
    assert why is not None
    assert str(k) in why, "the reason has to say how many, or a reader cannot check it"


def test_a_direction_per_layer_is_refused_even_with_one_direction_each():
    """THE CASE MOST EASILY GOT WRONG. K is 1, so a check on K alone passes it, and there is still
    no single direction: each layer got its own, so a profile across positions would be a profile
    against a different ruler at every point.
    """
    why = residualleak.unmeasurable_because("per_layer", 1)
    assert why is not None
    assert "per_layer" in why


def test_the_refusal_names_the_recipe_rather_than_apologising():
    why = residualleak.unmeasurable_because("single", 4)
    assert "4 directions" in why
    assert "different quantity" in why


# ── the block, which is always present in one of three states ────────────────────────

def test_nobody_asked_and_asked_but_not_measured_are_different_states():
    """The distinction `_capability_block` exists to keep, kept here too. An artefact that spells
    them the same way hides the second one, which is the only one that needs reading.
    """
    quiet = residualleak.leak_block(requested=False)
    asked = residualleak.leak_block(refused_because="no single direction", prompts=32)
    assert quiet["state"] == residualleak.LEAK_NOT_REQUESTED
    assert asked["state"] == residualleak.LEAK_REQUESTED_BUT_NOT_MEASURED
    assert quiet["state"] != asked["state"]
    assert quiet["why_not"] is None and asked["why_not"] == "no single direction"
    assert quiet["measured"] is False and asked["measured"] is False


def test_a_measured_block_carries_the_profile_and_the_mean():
    block = residualleak.leak_block(_Report())
    assert block["state"] == residualleak.LEAK_MEASURED
    assert block["measured"] is True
    assert len(block["per_position"]) == 4
    assert block["mean"] == pytest.approx(1.375e-8)
    assert block["positions"] == 4
    assert block["probe_prompts"] == 32


def test_the_output_figure_is_the_post_norm_one_and_never_the_pre_norm_one_at_the_output():
    """THE MISREADING THE WHOLE MODULE IS SHAPED TO PREVENT. With a perfect ablation the pre-norm
    direction still reads about 1% at the output, because a diagonal map does not preserve
    orthogonality to a fixed vector. Publishing that figure would be this project reporting a
    basis change as a defect, which its own docstring calls worse than no tool.
    """
    block = residualleak.leak_block(_Report())
    assert block["output_along_post_norm_direction"] == pytest.approx(3e-8)
    assert "along_pre_norm_direction" not in block
    assert 0.0096 not in block.values()


def test_an_unusable_norm_omits_the_output_figure_and_keeps_the_reason():
    block = residualleak.leak_block(_Report(output=_Output(post=None, refused="no diagonal found")))
    assert block["output_along_post_norm_direction"] is None
    assert block["output_basis_refused"] == "no diagonal found"


def test_no_final_norm_at_all_is_not_the_same_as_an_unusable_one():
    block = residualleak.leak_block(_Report(output=None))
    assert block["output_along_post_norm_direction"] is None
    assert block["output_basis_refused"] is None


def test_degradations_travel_with_the_figure():
    block = residualleak.leak_block(_Report(warnings=("no final norm was found",)))
    assert block["warnings"] == ["no final norm was found"]


def test_a_block_cannot_claim_the_figure_both_exists_and_does_not():
    with pytest.raises(ValueError, match="exactly one"):
        residualleak.leak_block()
    with pytest.raises(ValueError, match="exactly one"):
        residualleak.leak_block(_Report(), refused_because="and also no")


# ── and through the abliterator's method, which needs no GPU to drive ────────────────

class _Run:
    """The parts of the abliterator `leak_report` reads, and nothing else."""

    def __init__(self, *, direction=None, raises=None, report=None, leak_prompts=32, n_eval=40):
        self.args = SimpleNamespace(leak_prompts=leak_prompts)
        self.bad_eval = [f"harmful prompt {i}" for i in range(n_eval)]
        self.model, self.tok = object(), object()
        self.said = []
        self.log = self.said.append
        self._direction = direction if direction is not None else [1.0, 0.0]
        self._raises, self._report = raises, report
        self.seen = {}

    def active_dirs(self, idx, k):
        self.seen["active_dirs"] = (idx, k)
        return [self._direction]

    leak_report = cli.Abliterator.leak_report


def _measured(run, monkeypatch, report=None):
    def _fake(model, tok, prompts, direction, *, log=print):
        run.seen["prompts"] = list(prompts)
        run.seen["direction"] = direction
        if run._raises is not None:
            raise run._raises
        return report or _Report()

    monkeypatch.setattr(residualleak, "measure_leak", _fake)
    return run.leak_report(1, "single")


def test_the_direction_comes_from_the_function_the_bake_used(monkeypatch):
    """NOT FROM A SECOND RESOLUTION OF THE SAME PARAMETERS. A second resolution is a second thing
    that can drift, and a leak figure measured against a direction the bake did not apply would be
    a perfectly real number about nothing that happened. `cli.py` and `residualleak.py` have
    already drifted one such list between them, which is why this is asserted rather than assumed.
    """
    run = _Run(direction=[0.6, 0.8])
    block = _measured(run, monkeypatch)
    assert run.seen["active_dirs"] == (0, 1), "index 0 and exactly one direction"
    assert run.seen["direction"] == [0.6, 0.8]
    assert block["state"] == residualleak.LEAK_MEASURED


def test_an_unmeasurable_recipe_never_reaches_the_instrument(monkeypatch):
    """The refusal is checked before a forward pass is spent, and before a direction is picked."""
    run = _Run()
    monkeypatch.setattr(residualleak, "measure_leak",
                        lambda *a, **k: pytest.fail("the instrument was called anyway"))
    block = run.leak_report(3, "single")
    assert block["state"] == residualleak.LEAK_REQUESTED_BUT_NOT_MEASURED
    assert "active_dirs" not in run.seen
    assert any("residual leak: not reported" in line for line in run.said)


def test_the_prompt_count_is_honoured_and_zero_means_all_of_them(monkeypatch):
    run = _Run(leak_prompts=8, n_eval=40)
    _measured(run, monkeypatch)
    assert len(run.seen["prompts"]) == 8

    everything = _Run(leak_prompts=0, n_eval=40)
    _measured(everything, monkeypatch)
    assert len(everything.seen["prompts"]) == 40


def test_a_failed_measurement_costs_the_figure_and_never_the_record(monkeypatch):
    """THE ORDER OF PRIORITIES, asserted because it is the whole reason the except is broad.

    By the time this runs the weights are on disk and `abliteration.json` is not yet written, so
    an exception escaping here would end a finished run with a traceback and leave a checkpoint
    with no provenance at all. The same defect shape `restore_device_after_save` was given a guard
    for, one measurement later in the same function.
    """
    run = _Run(raises=ValueError("3 probe prompt(s), below the floor of 4"))
    block = _measured(run, monkeypatch)
    assert block["state"] == residualleak.LEAK_REQUESTED_BUT_NOT_MEASURED
    assert "below the floor" in block["why_not"]
    assert "ValueError" in block["why_not"]
    assert "saved weights are unaffected" in block["why_not"]
    assert block["probe_prompts"] == 32


def test_the_figure_is_said_out_loud_and_not_only_filed(monkeypatch):
    """A number that reaches only a field is invisible to somebody watching a terminal, and the
    figure they will quote is the one they watched arrive.
    """
    run = _Run()
    _measured(run, monkeypatch)
    said = "\n".join(run.said)
    assert "residual leak over 32 prompts" in said
    assert "at the output" in said


def test_an_output_basis_refusal_is_said_out_loud_too(monkeypatch):
    run = _Run()
    _measured(run, monkeypatch,
              report=_Report(output=_Output(post=None, refused="the norm is not diagonal")))
    assert any("no output figure: the norm is not diagonal" in line for line in run.said)


# ── and into the artefact, which is what somebody else actually reads ────────────────

BPR = (12, 0.9, 0.1, 4, 12, 0.7, 0.0, 4)


def _record(leak=None, **over):
    from tests.test_abliteration_record import _args, _run

    post = dict(refusals=0.02, heretic=0.01, broken=0.0, kl=0.13)
    post.update(over)
    return cli.build_abliteration_record(_run(), _args(), BPR, 1, "single", 0,
                                         base_ref=0.39, post=post, leak=leak)


def test_the_record_always_carries_a_leak_block():
    """Always present, in one of three states. An absent key reads as whoever wrote the file
    forgetting rather than as a statement about the run.
    """
    assert _record()["residual_leak"]["state"] == residualleak.LEAK_NOT_REQUESTED


def test_a_measured_leak_reaches_the_record():
    rec = _record(leak=residualleak.leak_block(_Report()))
    assert rec["residual_leak"]["measured"] is True
    assert rec["residual_leak"]["mean"] == pytest.approx(1.375e-8)


def test_a_caller_written_before_this_field_existed_still_builds_a_record():
    """The same courtesy the capability fields were given when they were added."""
    rec = cli.build_abliteration_record(
        __import__("tests.test_abliteration_record", fromlist=["_run"])._run(),
        __import__("tests.test_abliteration_record", fromlist=["_args"])._args(),
        BPR, 1, "single", 0, base_ref=0.39,
        post=dict(refusals=0.02, heretic=0.01, broken=0.0, kl=0.13))
    assert rec["residual_leak"]["state"] == residualleak.LEAK_NOT_REQUESTED


def test_the_flag_exists_and_is_off_by_default():
    """A measurement nobody asked for must not be taken, and a flag the help text describes has to
    be one the parser actually declares. Both halves have shipped broken here before.
    """
    from senbonzakura.parser import build_parser

    dests = {a.dest for a in build_parser()._actions}
    assert "leak_report" in dests
    assert "leak_prompts" in dests
    a = build_parser().parse_args([])
    assert a.leak_report is False
    assert a.leak_prompts == 32


def test_the_count_is_not_named_prompts_at_any_depth():
    """THE ONE CONTROL BETWEEN A HARMFUL PROMPT SET AND A PUBLIC PUSH, and it is never widened.

    `tools/ci/check_prompt_artefacts.py` refuses a committed artefact carrying a key named
    `prompts` at any depth, because that is how such a set reaches a public repository wearing a
    measurement's name. The first draft of the leak block called its count `prompts`, which would
    have made `abliteration.json` uncommittable, and that gate's own test caught it. Pinned here
    too so the next person renaming a field meets the reason rather than the failure.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "ci"))
    import check_prompt_artefacts as gate

    def keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from keys(v)

    for block in (residualleak.leak_block(requested=False),
                  residualleak.leak_block(refused_because="no single direction", prompts=32),
                  residualleak.leak_block(_Report())):
        assert not ({k.lower() for k in keys(block)} & gate.BANNED_KEYS)
    assert "probe_prompts" not in gate.BANNED_KEYS, (
        "if probe_prompts is ever banned too, rename the field again rather than relaxing the gate")
