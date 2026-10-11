# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A device this machine cannot use is refused before the weights are fetched, for every command.

`cli._preflight_device` has refused `--device cuda` on a machine with no card since 2026-09-16, and
its own docstring names the lesson: the failure is the one "`entry.py` goes to some trouble to
prevent for a missing import, arriving through the one input nobody checked". It was wired into the
abliterate path and the guided mode. **Twelve other commands take the same flag with the same
`cuda` default and none of them checked it.**

So `senbonzakura compass --model Qwen/Qwen3-1.7B ...` fetched 3.8 GB, loaded it, began scoring, and
died in `margin.py` on `.to(device)` under fourteen frames of torch. Found by the constrained-user
journey of 2026-10-11, reading as somebody on a metered connection for whom that download is the
entire cost of finding out.

**THE FIRST FIX WAS IN THE WRONG PLACE, and the way it failed is worth keeping.** The check went
into each command's `main`, immediately after `parse_args`. That is earlier, and earlier was wrong:
a surface check preempts every refusal the command does for itself, so eighteen tests asserting a
specific early refusal (identical arms, a contradicted boundary, a misread ruler) received the
device refusal instead. Worse, they received it only on a machine with no card, **which is CI**, so
the suite's behaviour would have come to depend on the test host's hardware.

The check belongs beside `_refuse_without_the_checker` at the top of the shared loader. Everything
expensive is downstream of it, `doctor` never reaches it, a command's own pre-flights all run
before it, and a test that asserts a refusal before the model loads never reaches it either. That
is the same answer this project arrived at for the missing-dependency pre-flight, for the same
reason: predicting which commands need the check is the wrong question.
"""

import pathlib
from unittest import mock

import pytest

from senbonzakura import argresolve

# ── the rule: the loader refuses, so every command inherits it ───────────────────

def test_the_loader_refuses_before_it_reaches_transformers():
    """The ordering claim, driven rather than asserted against the source.

    `from_pretrained` is replaced with something that fails loudly. A late check sees that
    explosion; an early one never reaches it. The tokenizer load is stubbed too, because it also
    touches the network and runs first.
    """
    from senbonzakura import cli

    def must_not_be_reached(*a, **k):
        raise AssertionError("the model fetch ran, so the device check was too late to save the "
                             "download it exists to save")

    with mock.patch.object(cli, "load_tokenizer", return_value=object()), \
         mock.patch("transformers.AutoModelForCausalLM.from_pretrained", must_not_be_reached), \
         mock.patch("torch.cuda.is_available", return_value=False), pytest.raises(SystemExit) as e:
        cli.load_model_and_tokenizer("Qwen/Qwen3-1.7B", device="cuda")
    assert "no usable CUDA device" in str(e.value)
    assert "--device cpu" in str(e.value), "the refusal has to say what to pass instead"


def test_the_check_sits_after_the_security_check_and_before_the_weights():
    """Read off the source, because the POSITION is the design decision this file records.

    Two orderings were wrong before this one, and the test encodes both lessons.

    It must not be at the command surfaces: that preempts every refusal a command does for itself,
    and only on a machine with no card, so the suite's behaviour follows the test host's hardware.

    And it must not precede the checkpoint-index check. Put first in the loader, it stood in front
    of the PYSEC-2026-3804 refusal, so a hostile checkpoint went unexamined whenever the device
    string happened to be wrong too. **A convenience check must never be able to preempt a security
    one.** The weights are downstream either way, so nothing is lost by going second.
    """
    from senbonzakura import cli
    src = pathlib.Path(cli.__file__).read_text(encoding="utf-8")
    # The whole remainder, not a fixed window: this function carries several hundred lines of
    # comment and a 3000-character window fell short of its own `from_pretrained`.
    head = src.split("def load_model_and_tokenizer", 1)[1]
    assert "argresolve.refuse_unusable_device(device)" in head, (
        "the loader no longer refuses an unusable device, so every command that loads a model will "
        "pay for the download before finding out")

    device_at = head.index("argresolve.refuse_unusable_device(device)")
    index_at = head.index("refuse_unsafe_hub_index")
    fetch_at = head.index("from_pretrained")

    assert index_at < device_at, (
        "the device check now precedes the checkpoint-index check, so a hostile checkpoint goes "
        "unexamined when the device string is also wrong. Security first")
    assert device_at < fetch_at, (
        "the device check now follows the model fetch, which is the download it exists to save")


def test_a_command_surface_does_not_preempt_its_own_preflights():
    """The regression this file exists to prevent, stated as a rule about where the call is NOT.

    Eighteen tests failed when the call lived in each `main`. The modules below all run their own
    refusals first and must keep that order.
    """
    src_dir = pathlib.Path(argresolve.__file__).parent
    for module in ["margin", "drift", "score", "jailbreak", "multiturn", "tamper", "leak",
                   "capability", "coherence"]:
        text = (src_dir / f"{module}.py").read_text(encoding="utf-8")
        assert "refuse_unusable_device" not in text, (
            f"`{module}` calls the device check at its own surface again. That preempts the "
            f"refusals it does for itself, and it does so only on a machine with no card, so the "
            f"suite starts depending on the test host's hardware. The loader is the place")


def test_compass_still_refuses_end_to_end():
    """The user-visible behaviour, through the real entry point and the real parser.

    SKIPPED WITHOUT THE PACKED TRACK, which is CI. `src/senbonzakura/data/*.bin` are built at
    release time and deliberately not committed, because they hold harmful prompts, so
    `--track default` refuses before the loader is reached and this test asserted the wrong
    refusal. `bundled.is_available()` is the helper the rest of the suite settled on for exactly
    this, and the first version of this file was written without it on the same day the lesson was
    recorded elsewhere.
    """
    from senbonzakura import bundled, margin

    if not bundled.is_available():
        pytest.skip("no packed track in this tree; the loader is unreachable without one")

    def must_not_be_reached(*a, **k):
        raise AssertionError("compass reached the loader's model fetch")

    with mock.patch("transformers.AutoModelForCausalLM.from_pretrained", must_not_be_reached), \
         mock.patch("torch.cuda.is_available", return_value=False), pytest.raises(SystemExit) as e:
        margin.main(["--model", "Qwen/Qwen3-1.7B", "--harmful", "default/bad_eval_ds",
                     "--harmless", "default/good_ds", "--track", "default",
                     "--device", "cuda", "--n", "4", "--out", "unused.json"])
    assert "no usable CUDA device" in str(e.value)


# ── the shared helper's own behaviour ────────────────────────────────────────────

@pytest.mark.parametrize("device", ["cpu", "CPU", " cpu ", "cpu:0", "", None])
def test_a_processor_or_an_unset_device_is_always_allowed(device):
    """No machine can fail this, so it must never raise: it is the fallback the refusal names."""
    assert argresolve.refuse_unusable_device(device) is not None


def test_an_unknown_device_name_is_refused_with_the_list(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    with pytest.raises(SystemExit) as e:
        argresolve.refuse_unusable_device("banana")
    assert "not a device this tool knows" in str(e.value)
    assert "--device cuda" in str(e.value)


def test_a_card_that_is_present_is_not_refused(monkeypatch):
    """The other direction. A check that refuses everything would pass every test above."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert argresolve.refuse_unusable_device("cuda") == "cuda"
    assert argresolve.refuse_unusable_device("cuda:1") == "cuda:1"


def test_the_refusal_names_doctor_and_setup(monkeypatch):
    """Both ways out, because the honest answer depends on whether the machine has a card at all."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(SystemExit) as e:
        argresolve.refuse_unusable_device("cuda")
    text = str(e.value)
    assert "doctor" in text
    assert "setup --apply" in text


def test_the_wrapper_the_abliterate_path_calls_still_delegates():
    """`cli._preflight_device` is monkeypatched by several tests, so the name has to survive."""
    from senbonzakura import cli

    class Args:
        device = "cpu"

    assert cli._preflight_device(Args()) == "cpu"


def test_help_does_not_pay_for_torch():
    """`argresolve` is on the `--help` path, which is measured, so torch is imported inside.

    `--help` went from 2.80s to 0.06s on 2026-09-02 by keeping heavy imports off this path, and a
    module-level `import torch` here would quietly undo it.
    """
    text = pathlib.Path(argresolve.__file__).read_text(encoding="utf-8")
    head = text.split("def refuse_unusable_device", 1)[0]
    assert "import torch" not in head, (
        "torch is imported at module scope in argresolve, which is on the --help path")
