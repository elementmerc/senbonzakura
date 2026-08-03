"""Tests for the axis probe (tools/axis_probe.py).

The probe exists to answer one question: when a layer keeps a single refusal direction, is that
because the model has one, or because a constant was set above the second one? Its verdict is the
whole output, so the tests are mostly about the boundary between the two readings and about the
probe not inventing a configuration the real tool never runs.
"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from senbonzakura import cli

_SPEC = importlib.util.spec_from_file_location(
    "axis_probe", Path(__file__).resolve().parent.parent / "tools" / "axis_probe.py")
probe = importlib.util.module_from_spec(_SPEC)
sys.modules["axis_probe"] = probe
_SPEC.loader.exec_module(probe)


def _fake(seps, *, threshold=None):
    """An object shaped like the parts of an Abliterator that `summarise` reads."""
    flat = [d for layer in seps for d in layer]
    below = [d for d in flat if d < (threshold or cli.MIN_AXIS_SEPARATION)]
    return SimpleNamespace(
        args=SimpleNamespace(model="m", track="t", dir_prompts=128, good_ds=None),
        H=8, NL=2, KMAX=8,
        dirs_per_layer=[1] * len(seps),
        axis_separations=seps,
        best_rejected_separation=max(below) if below else None)


# ── the verdict, which is the entire point of the tool ────────────────────────────────
def test_an_all_zero_result_is_reported_as_a_broken_instrument():
    """What every probe actually returned on 2026-08-03, on two models and three corpora.

    A separation of zero on every axis is not a small measurement. It is what the geometry
    forces: candidates are orthogonalised against a basis spanning both class means, and the
    statistic is a difference of class means. Reading that as "the model has one direction" is
    the mistake this branch exists to stop anyone making again.
    """
    r = probe.summarise(_fake([[0.0, 1e-8], [0.0, 3e-9]]))
    assert r["filter_is_unsatisfiable"] is True
    assert "cannot be passed at any positive threshold" in r["verdict"]
    assert "Nothing here measures direction count" in r["verdict"]


def test_the_zero_criterion_covers_the_residue_a_real_model_actually_produced():
    """1.07e-5, measured across 3,556 axes on Qwen3-1.7B.

    The first version of this criterion was an absolute 1e-6 taken from a synthetic case that
    produced ~1e-8, so it reported "not broken" about the very measurement it was written for.
    A constant calibrated on one dataset and never checked against another is exactly the
    failure the filter itself represents, which is why the criterion is now relative.
    """
    assert probe.summarise(_fake([[1.07e-5]]))["filter_is_unsatisfiable"] is True
    # And a separation small enough to be uninteresting is still a measurement, not residue.
    assert probe.summarise(_fake([[0.01]]))["filter_is_unsatisfiable"] is False


def test_the_probe_and_the_extractor_cannot_disagree_about_what_zero_means():
    """Two copies of a constant is how the prompt renderer drifted into three."""
    assert probe.STRUCTURAL_ZERO == cli.MIN_AXIS_SEPARATION * cli.STRUCTURAL_ZERO_FRACTION


def test_a_small_but_real_separation_is_not_mistaken_for_the_broken_case():
    """The boundary matters: 0.02 is a finding about the data, 1e-8 is a finding about the code."""
    r = probe.summarise(_fake([[0.02], [0.01]]))
    assert r["filter_is_unsatisfiable"] is False
    assert "absent, not filtered" in r["verdict"]


def test_no_axes_measured_is_not_called_broken():
    """A K=1 run measures nothing, and nothing is not evidence of a broken filter."""
    assert probe.summarise(_fake([[], []]))["filter_is_unsatisfiable"] is False



def test_a_near_miss_blames_the_threshold():
    """The reading that would mean the headline claim was suppressed by a constant."""
    r = probe.summarise(_fake([[cli.MIN_AXIS_SEPARATION - 0.01], [0.1]]))
    assert "the constant decided the direction count" in r["verdict"]
    assert r["best_rejected_separation"] == pytest.approx(cli.MIN_AXIS_SEPARATION - 0.01)


def test_a_clear_rejection_blames_the_model_or_the_corpus():
    r = probe.summarise(_fake([[0.05], [0.02]]))
    assert "absent, not filtered" in r["verdict"]


def test_nothing_rejected_says_the_threshold_did_not_bind():
    r = probe.summarise(_fake([[2.0], [1.4]]))
    assert r["best_rejected_separation"] is None
    assert "did not bind" in r["verdict"]


def test_the_boundary_between_the_two_readings_is_where_it_claims_to_be():
    """80% of the threshold. Named here so moving it has to move a test too."""
    edge = cli.MIN_AXIS_SEPARATION * 0.8
    assert "constant decided" in probe.summarise(_fake([[edge]]))["verdict"]
    assert "absent, not filtered" in probe.summarise(_fake([[edge - 1e-6]]))["verdict"]


def test_no_axes_at_all_is_not_a_crash():
    """A K=1 run measures no candidates, and a probe that dies on it is useless for comparison."""
    r = probe.summarise(_fake([[], []]))
    assert r["axes_measured"] == 0
    assert r["max_separation_any_axis"] is None
    assert r["best_rejected_separation"] is None


def test_the_harmless_side_override_is_recorded_even_when_unset():
    """Two probes that differ only in the harmless side must not produce identical records."""
    r = probe.summarise(_fake([[0.2]]))
    assert "good_ds_override" in r and r["good_ds_override"] is None


def test_the_counts_describe_the_separations_they_ship_with():
    r = probe.summarise(_fake([[0.9, 0.3], [0.2]]))
    assert r["axes_measured"] == 3
    assert r["axes_rejected"] == 2
    assert r["max_separation_any_axis"] == pytest.approx(0.9)
    assert r["axis_separation_threshold"] == cli.MIN_AXIS_SEPARATION


# ── the arguments come from the real parser ───────────────────────────────────────────
def test_the_probe_inherits_the_tools_own_defaults():
    """A probe with its own defaults measures a configuration nobody runs.

    Hand-written namespaces drifting from `build_parser()` is a failure this repository has
    already paid for once, so the probe builds its args through the parser and this asserts it.
    """
    _, args = probe.build_args(["--model", "some/model"])
    reference = cli.build_parser().parse_args(["--model", "some/model"])
    ignore = {"model", "track", "device", "max_directions", "dir_prompts"}
    for key, value in vars(reference).items():
        if key not in ignore:
            assert getattr(args, key) == value, f"{key} drifted from the parser default"


def test_the_probe_pursues_more_axes_than_a_real_run_would_apply():
    """The question is what is there, not what would be used, so the default K is deliberately high."""
    _, args = probe.build_args(["--model", "m"])
    assert args.max_directions > cli.build_parser().parse_args(["--model", "m"]).max_directions


def test_overrides_reach_the_parser():
    own, args = probe.build_args(
        ["--model", "m", "--track", "/tmp/tk", "--good-ds", "/tmp/g", "--device", "cpu",
         "--max-directions", "5", "--dir-prompts", "16", "--label", "run-a"])
    assert (args.model, args.track, args.good_ds) == ("m", "/tmp/tk", "/tmp/g")
    assert (args.device, args.max_directions, args.dir_prompts) == ("cpu", 5, 16)
    assert own.label == "run-a"


def test_the_optional_flags_are_off_unless_asked_for():
    _, plain = probe.build_args(["--model", "m"])
    assert not plain.load_in_4bit and not plain.trust_remote_code
    _, both = probe.build_args(["--model", "m", "--load-in-4bit", "--trust-remote-code"])
    assert both.load_in_4bit and both.trust_remote_code


# ── end to end against the tiny fixture ───────────────────────────────────────────────
def test_the_probe_runs_and_writes_a_record(base_args, tiny_model, tiny_tok, track,
                                            tmp_path, monkeypatch, capsys):
    """Drives the real extractor on the fixture model, which is what it does on a real one."""
    out = tmp_path / "probe.json"
    real = cli.Abliterator
    monkeypatch.setattr(cli, "Abliterator",
                        lambda args, log: real(args, log, model=tiny_model, tok=tiny_tok))
    rc = probe.main(["--model", "fixture", "--track", track, "--device", "cpu",
                     "--max-directions", "4", "--dir-prompts", "8", "--out", str(out)])

    assert rc == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["directions_per_layer"] and record["verdict"]
    assert record["axis_separation_threshold"] == cli.MIN_AXIS_SEPARATION
    assert len(record["axis_separations"]) == len(record["directions_per_layer"])
    assert "axis probe" in capsys.readouterr().out


def test_the_record_is_written_atomically(base_args, tiny_model, tiny_tok, track,
                                          tmp_path, monkeypatch):
    """An interrupted probe must not leave a half-written record that reads as a result."""
    out = tmp_path / "probe.json"
    real = cli.Abliterator
    monkeypatch.setattr(cli, "Abliterator",
                        lambda args, log: real(args, log, model=tiny_model, tok=tiny_tok))
    monkeypatch.setattr(probe.json, "dump", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt))

    with pytest.raises(KeyboardInterrupt):
        probe.main(["--model", "fixture", "--track", track, "--device", "cpu",
                    "--max-directions", "4", "--dir-prompts", "8", "--out", str(out)])
    assert not out.exists()
    assert not list(tmp_path.glob("*.part"))
