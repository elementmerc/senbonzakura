# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""An instrument whose whole output is "these two agree" has to be able to say no.

READ THE SECOND PARAGRAPH BEFORE ADDING A FILE TO `tools/research/`.

A comparison that always passes and a comparison that is correct look identical from the outside,
every time, for ever. The only way to tell them apart is to break the comparison on purpose and
watch the instrument notice. `parity_check.py` has that, properly, and its own comments record the
lesson the hard way: the first version of its `drop` control silently removed nothing, so the gate
correctly reported no change and the control reported the gate as toothless. The control was the
broken part.

THIS FILE IS THE GATE THAT KEEPS THAT TRUE OF THE NEXT ONE. Every module under `tools/research/`
is classified here, as either an instrument that needs a control or one that does not, with the
reason. A new file is in neither list and fails this test until somebody says which it is. That is
the point: the question gets asked once per instrument, by the person adding it, rather than
rediscovered by a panel reviewer two months later.

WHAT THIS DOES NOT DO. It does not run the instruments. Most of them need torch, a card and a
model, and `tests/test_parity_check.py` already drives the one whose comparison is a pure function
over tensors through all three of its controls, in the suite, which is to say in CI on every push.
The premise that the controls "never run in CI" was checked and is false: it came from grepping
`ci.yml` for the script's name, and the suite is how it runs.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "tools" / "research"

#: Instruments whose entire output is an agreement, with the control each one carries.
#:
#: The value is a token that must appear in the file, chosen to be the control itself rather than
#: the word "control", because a comment mentioning controls is not a control.
NEEDS_A_CONTROL = {
    "parity_check.py": (
        "its whole output is whether two runs describe the same subspace",
        "--control"),
    "measure_separation.py": (
        ("its whole output is whether a statistic separates two labelled groups, which is the same "
         "shape of claim: a separation that is really an artefact of the split looks identical"),
        "SHUFFLED control"),
    "derive_refusal_markers.py": (
        ("its whole output is Cohen's kappa between a derived marker list and our semantic refusal "
         "metric, so it is an agreement by construction. And the control was not a formality: with "
         "the labels shuffled it still reaches an IN-SAMPLE kappa of 0.5923, because a greedy "
         "search over every n-gram in 259 replies can find phrases that separate any labelling at "
         "all. Without the control, the real run's in-sample 0.9920 reads as a result when the "
         "floor for pure noise is 0.59"),
        "--control shuffled"),
    "matched_leak_strength.py": (
        ("its whole output is whether two models were edited to the same place, which is an "
         "agreement by construction. The expensive way to be wrong is a table of two matched "
         "strengths where one arm never reached the target, with the difference between them then "
         "attributed to the models rather than to the arm that did not converge"),
        "caveat_if_unconverged"),
    "decoy_injection.py": (
        ("its whole output is whether `validate` agrees with a ground truth we planted, so it is "
         "an agreement against a known answer. The control is the arm that says nothing: an alpha "
         "too small leaves the estimator on refusal and the arm is an undefended model, an alpha "
         "too large breaks the model and a broken model defeats the estimator too. Without that, "
         "a sweep where no injection worked reads as a clean bill of health for the detector, "
         "which is the one conclusion the data cannot support"),
        "uninformative"),
}

#: Everything else under `tools/research/`, each with why a breaking control would mean nothing
#: for it. Being on this list is a claim somebody made, not an absence of one.
NO_AGREEMENT_TO_BREAK = {
    "audit_flags.py": "enumerates declared flags; its output is a list, not a verdict",
    "experiments.py": "registers the open experiments and derives, per arm, whether this machine "
                      "can run it. Its output is a table and a generated batch script, and the "
                      "one claim it makes about the world is arithmetic over a declared model "
                      "size against a card size read from the machine. Nothing in it compares two "
                      "measurements, so there is no agreement a control could break. Its own "
                      "guard is that no verdict is storable: a test moves every arm between "
                      "verdicts by changing only the card size",
    "audit_unreached.py": "enumerates public names with no caller; its output is a list for a "
                          "person to rule on, and it says so in its own output. It cannot tell a "
                          "dead knob from a piece built ahead of its caller, which is why it "
                          "prints rather than passing or failing, so there is no verdict to "
                          "break. Its own validation is a known-present case, run against the "
                          "commit before a wiring landed and again after",
    "audit_token_positions.py": "reports which positions a tokeniser produces; descriptive",
    "axis_probe.py": "reports a score per axis; nothing in its output is an agreement",
    "direction_validation.py": "reports properties of one basis, not a comparison of two",
    "expert_layout.py": "prints the expert layout of a checkpoint; descriptive",
    "fetch_architecture_corpus.py": "collects and normalises published configs and tensor "
                                    "inventories; its output is a corpus and a vocabulary, not "
                                    "a verdict. The one thing it could get silently wrong is "
                                    "reporting a model as reached when it was partly fetched, "
                                    "which it records as partial with the reason rather than "
                                    "deciding",
    "gpu_lock.py": "an interlock, not an instrument. Its output is whether one job may use the "
                   "card, and it has no comparison to break. The agreement it DOES depend on, "
                   "that a lock file and the compute-app list are two independent facts, is "
                   "covered by a test that gives it a free lock over a busy card and requires a "
                   "refusal",
    "layer_read_spike.py": "a timing spike; its output is a duration",
    "leak_sweep.py": "reports a distribution of row lengths; its synthetic sweep is an input "
                     "rather than a control over a verdict",
    "mutate.py": "the mutation harness, whose whole job is to break things on purpose. It is the "
                 "control, and a control over a control is not a thing",
    "pinned_memory_spike.py": "a timing spike; its output is a duration",
    "rdo.py": "an optimiser; its output is a chosen configuration and a score, not an agreement",
    "report_bands.py": "formats bands for a report; it computes no verdict",
    "shard_spike.py": "a timing and memory spike; its output is a measurement",
    "check_glm_lite_convert.py": "asks whether ONE checkpoint converts to a GGUF that llama.cpp "
                                 "will load, and whether its declared block count matches the "
                                 "tensors it actually saved. Both are self-consistency properties "
                                 "of a single artefact rather than an agreement between two, so "
                                 "there is no comparison for a control to break",
}


def _modules():
    return sorted(p.name for p in RESEARCH.glob("*.py") if not p.name.startswith("_"))


def test_every_research_module_is_classified():
    """The whole gate. A new instrument is in neither list and fails here until it is in one.

    Deliberately not a vocabulary scan over the source. "Does this file talk about agreement"
    cannot be answered by grepping, and a guard that answers a nearby question and reports with
    confidence is this project's most repeated defect. A complete partition can only be satisfied
    by somebody reading the file.
    """
    classified = set(NEEDS_A_CONTROL) | set(NO_AGREEMENT_TO_BREAK)
    found = set(_modules())
    unclassified = sorted(found - classified)
    assert not unclassified, (
        f"these modules under tools/research/ are in neither list in this file: {unclassified}.\n"
        f"  If the module's output is whether two things agree, give it a control that breaks the "
        f"comparison on purpose and add it to NEEDS_A_CONTROL.\n"
        f"  If it is not that kind of instrument, add it to NO_AGREEMENT_TO_BREAK with the reason. "
        f"A comparison that always passes and a comparison that is correct look identical from "
        f"the outside, every time, for ever.")
    stale = sorted(classified - found)
    assert not stale, (
        f"these modules are classified here and no longer exist: {stale}. A register naming files "
        f"that are gone is a register nobody trusts.")


def test_the_scan_actually_finds_modules():
    """Without this, a moved directory makes the gate above pass over an empty set."""
    assert len(_modules()) > 5, _modules()


@pytest.mark.parametrize("name", sorted(NEEDS_A_CONTROL))
def test_each_agreement_instrument_carries_its_control(name):
    reason, token = NEEDS_A_CONTROL[name]
    source = (RESEARCH / name).read_text(encoding="utf-8")
    assert token in source, (
        f"{name} is registered as an agreement instrument ({reason}) and its control, {token!r}, "
        f"is not in the file. An instrument that cannot be made to fail has not been shown to "
        f"work.")


def test_the_parity_controls_are_each_reachable_from_the_command_line():
    """Three modes, read off the parser rather than from the documentation.

    `rotate` is the one whose correct outcome is agreement, because a rotated basis is one
    subspace described twice, and the other two must be caught. A gate offering one control is a
    gate with one way of being wrong that it has looked at.
    """
    source = (RESEARCH / "parity_check.py").read_text(encoding="utf-8")
    choices = None
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if not any(isinstance(a, ast.Constant) and a.value == "--control" for a in node.args):
            continue
        for kw in node.keywords:
            if kw.arg == "choices":
                choices = [e.value for e in kw.value.elts]
    assert choices is not None, "parity_check.py declares no --control choices"
    assert set(choices) == {"rotate", "replace", "drop"}, choices


def test_the_controls_are_driven_by_the_suite_and_therefore_by_ci():
    """The claim the module docstring makes, asserted rather than left as prose.

    If `tests/test_parity_check.py` stops exercising the controls, the honest reading of this
    whole file changes, so the dependency is pinned here rather than assumed.
    """
    driver = (ROOT / "tests" / "test_parity_check.py").read_text(encoding="utf-8")
    for control in ("rotate", "replace", "drop"):
        assert f'"{control}"' in driver, (
            f"nothing in the suite exercises the {control!r} control, so it runs nowhere")
