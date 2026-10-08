# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Did the refusal direction leave the residual stream? On any model, including somebody else's.

WHAT THIS ANSWERS THAT NOTHING ELSE HERE DOES

Every other figure this tool reports is behavioural: refusal rate, harm recognition, fluency,
capability. Behaviour answers "did the model change". This answers the prior question, "is the
direction still in there", and it answers it with no judge, no sampling and not one generated
token, so it has nothing to validate a grader against and no run-to-run variance worth speaking of.

WHY IT IS A COMMAND AND NOT ONLY A FLAG

`abliterate --leak-report` already reports this for a model we just edited, which is the cheap case
because the direction is in hand by then. This is the other case, and it is the one that makes the
metric a ruler rather than a self-assessment: **point it at a checkpoint somebody else published and
it says whether their edit landed.** That works even on models this tool cannot edit at all, since
reading a model is just running it, so a 1.6 bit quantisation-aware model is measurable here and
uneditable upstream.

It extracts its own direction, which is the whole cost. Nothing writes a direction to disk, so
there is nothing to load.

THE CONTRAST IS THE PART THAT GOES WRONG, AND THE DEFAULT IS CHOSEN TO AVOID IT

A direction is only about refusal if the two prompt sets differ in refusal and not in their words.
This project has already withdrawn a figure for getting that wrong: a depth probe reported held-out
AUC 0.99 at layer 1 against `advbench` and `xstest-safe`, which was reading VOCABULARY, because
those two corpora differ lexically as well as behaviourally. The matched pair is `xstest-unsafe`
against `xstest-safe`, same source and same register, and that is the default here. Ask for an
unmatched pair and the run says what it is doing.

AND THE ONE CASE WHERE A GOOD FIGURE HERE MEANS THE OPPOSITE OF WHAT IT READS

What this measures is whether **the direction extracted here** has left the residual stream. That is
the same thing as refusal leaving the model only when the extracted direction carried refusal, and
there is now a published defence built to break that equality. Decoy Direction Optimization (arXiv
2609.16204) injects a high-magnitude feature orthogonal to refusal so that a contrastive estimator
finds the decoy; a run then ablates the decoy, reports a LOW leak, and leaves refusal exactly where
it was. AMRA (arXiv 2608.18093) attacks the extractor the same way by a different route.

So **a low figure beside an unchanged refusal rate is the signature of an edit that landed on a
decoy**, and the pair is what carries the information. Having no judge and no sampling makes this
figure free of grader error and run-to-run variance; it says nothing whatever about whether the
right direction was found. `validate` is the command for that question: leave-one-cluster-out
generalisation is what a decoy should fail while scoring well on magnitude.
"""
import json

from . import argresolve

DEFAULT_OUT = "leak.json"

#: The matched pair, same source and same register, differing in the thing being measured. See the
#: module docstring: the unmatched alternative produced a withdrawn figure, so this is a default
#: chosen against a known defect rather than by convenience.
DEFAULT_HARMFUL = "xstest-unsafe"
DEFAULT_HARMLESS = "xstest-safe"

#: Pairs whose two halves come from different sources, so a difference between them is partly a
#: difference in wording. Not refused, because somebody may want exactly this comparison, but never
#: run silently.
UNMATCHED = {
    ("advbench", "xstest-safe"): (
        "advbench and xstest-safe come from different sources and differ lexically as well as "
        "behaviourally. A depth probe in this project read AUC 0.99 at layer 1 from this pair and "
        "the figure was withdrawn: it was measuring vocabulary. The matched pair is "
        "xstest-unsafe against xstest-safe."),
}


def contrast_warning(harmful, harmless):
    """What is wrong with this pair of corpora, or None when they are matched."""
    direct = UNMATCHED.get((harmful, harmless))
    if direct:
        return direct
    if harmful == harmless:
        return (f"both arms are {harmful!r}, so the difference of means is zero by construction "
                f"and no direction can come out of it.")
    return None


def build_parser():
    from .parser import loader_parser

    ap = argresolve.ParserThatNamesUnknownFlags(
        allow_abbrev=False,
        prog="senbonzakura leak",
        description="Measure whether the refusal direction has left a model's residual stream. "
                    "Works on any checkpoint, including one this tool cannot edit, because "
                    "reading a model is just running it.",
        parents=[loader_parser(model_required=False)])
    ap.add_argument("model_positional", nargs="?", default=None, metavar="MODEL",
                    help="the model to measure, given without a flag. Equivalent to --model.")
    ap.add_argument("--harmful", default=DEFAULT_HARMFUL,
                    help=f"the corpus whose prompts a model should refuse (default: "
                         f"{DEFAULT_HARMFUL})")
    ap.add_argument("--harmless", default=DEFAULT_HARMLESS,
                    help=f"the matched corpus it should answer (default: {DEFAULT_HARMLESS}). The "
                         f"default pair shares a source and a register, so the difference between "
                         f"them is the behaviour rather than the wording")
    ap.add_argument("--n", type=argresolve.whole_number("--n", minimum=1), default=64,
                    help="prompts per arm for extracting the direction (default: 64)")
    ap.add_argument("--probe-n", dest="probe_n",
                    type=argresolve.whole_number("--probe-n", minimum=1), default=32,
                    help="prompts the leak profile is averaged over (default: 32). One forward "
                         "pass each with no generation, so this is seconds")
    # minimum=0 because position 0 is a real position, the first one. Not a bare `int`: a residual
    # position indexes a tensor, so `--at -1` is a perfectly good slice that silently means the last
    # position and reports a figure stamped with a position that was never asked for.
    ap.add_argument("--at", type=argresolve.whole_number("--at", minimum=0), default=None,
                    help="take the direction at this residual-stream position rather than where "
                         "the contrast is longest. Pass this when a separation has actually been "
                         "measured; the default rule is a norm and not a test")
    ap.add_argument("--out", default=None,
                    help=f"where the profile is written (default: ./{DEFAULT_OUT}). A default "
                         f"that already exists is refused rather than replaced")
    ap.add_argument("--label", default="", help="a name for this arm, recorded in the output")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)

    from .argresolve import pick_model, refuse_to_overwrite
    a.model = pick_model(a.model_positional, a.model, command="senbonzakura leak")
    if a.out is None:
        a.out = refuse_to_overwrite(DEFAULT_OUT, what="leak result")

    # BEFORE THE WEIGHTS, because a confounded contrast is a typo-grade mistake and should cost a
    # second rather than a model download and two forward passes per prompt.
    from . import corpora
    warning = contrast_warning(a.harmful, a.harmless)
    if warning and a.harmful == a.harmless:
        raise SystemExit(f"--harmful and --harmless: {warning}")

    try:
        harmful = corpora.load(a.harmful)
        harmless = corpora.load(a.harmless)
    except corpora.CorpusError as e:
        raise SystemExit(str(e)) from e

    if warning:
        # Said, not refused. Somebody may want this comparison, and the one thing that must not
        # happen is it running silently.
        print(f"  CONTRAST NOT MATCHED: {warning}")

    harmful = list(harmful)[:a.n]
    harmless = list(harmless)[:a.n]
    if not harmful or not harmless:
        raise SystemExit(
            f"one arm came back empty ({len(harmful)} harmful, {len(harmless)} harmless), so "
            f"there is no contrast to take a direction from.")

    from . import residualleak
    from .cli import load_model_and_tokenizer
    model, tok = load_model_and_tokenizer(
        a.model, device=a.device, load_in_4bit=a.load_in_4bit,
        trust_remote_code=a.trust_remote_code, chat_template=a.chat_template)

    print(f"leak: {a.label or a.model}")
    direction, position, norms, caveats = residualleak.contrast_direction(
        model, tok, harmful, harmless, at=a.at, log=print)

    probe = harmful[:a.probe_n]
    try:
        report = residualleak.measure_leak(model, tok, probe, direction, log=print)
    except ValueError as e:
        # The metric's own refusals are about a model it cannot measure, and they are better
        # messages than anything this layer could write, so they travel unchanged.
        raise SystemExit(f"leak: {e}") from e

    block = residualleak.leak_block(report)
    print(f"  profile over {report.positions} positions, mean {report.mean:.3e}, "
          f"in {report.basis}")
    out = report.output
    if out is not None and out.along_post_norm_direction is not None:
        print(f"  at the output, in the basis the final norm maps into: "
              f"{out.along_post_norm_direction:.3e}")
    elif out is not None:
        print(f"  no output figure: {out.refused_because}")
    for caveat in caveats:
        print(f"  CAVEAT: {caveat}")
    for note in report.warnings:
        print(f"  WARNING: {note}")

    from .crashsafe import atomic_write, provenance
    result = {
        "label": a.label or None,
        "model": a.model,
        "harmful": a.harmful,
        "harmless": a.harmless,
        "contrast_matched": warning is None,
        "contrast_warning": warning,
        "direction_position": position,
        # The evidence for the position, so a reader can see whether the choice was a peak or a
        # coin toss rather than taking the number on trust.
        "difference_norms": norms,
        "extraction_prompts": {"harmful": len(harmful), "harmless": len(harmless)},
        "caveats": list(caveats),
        "residual_leak": block,
        "provenance": provenance(device=a.device),
    }
    residualleak.stamp_report(result, report)
    with atomic_write(a.out) as f:
        json.dump(result, f, indent=2)
    print(f"  written to {a.out}")
    return 0


# AT THE END OF THE FILE, for the reason recorded at the bottom of `capability.py`: running a
# module as a script executes this before anything defined below it, so a block placed higher up
# makes `python -m senbonzakura.leak` fail on names that do not exist yet while the console script
# works. `tests/test_main_block_is_last.py` holds the shape.
if __name__ == "__main__":   # pragma: no cover
    from .entry import module_entry
    module_entry(main)
