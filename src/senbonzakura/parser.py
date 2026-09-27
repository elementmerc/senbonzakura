# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The command-line surface, in a module that costs nothing to import.

WHY IT IS NOT IN `cli.py`

`cli.py` imports torch, optuna and transformers at module level, which is 2.8 seconds and a
deep-learning stack. The parser needs none of them: measured with an AST walk, there is not one
statement in either function below that touches torch, optuna or transformers. They were slow
purely by living in the same file.

Everything a person does that is not a run goes through here and now costs about a hundredth of a
second: `--help`, `--version`, every argument error, and the guided mode's first screen. Only an
actual abliteration pays for torch, which is the one case where the cost buys something.

    senbonzakura --help          2.80s  ->  0.02s
    a bad flag (argparse exit 2) 2.80s  ->  0.02s

WHAT MAY BE IMPORTED HERE

`argparse`, the version, `metrics` and `separation` (constants, with no heavy imports of their
own), and `events` (json, os, sys, time). Nothing else, ever. A future `import torch` here
silently restores the defect, so the test suite measures the cost rather than trusting this
paragraph, and it measures it for everything on that list rather than for this file alone.
"""
from __future__ import annotations

import argparse

from . import methods as _methods  # constants only; imports nothing heavy
from . import separation as _separation  # constants only; imports nothing heavy, see its head
from ._version import __version__
from .lengthsweep import DEFAULT_BUDGET as _DEFAULT_BUDGET  # constants only; pulls only .metrics
from .lengthsweep import VISIBILITY_FLOOR as _VISIBILITY_FLOOR
from .metrics import KL_CEIL, KL_TARGET


def _capability_tasks():
    """The grading tasks, read lazily so this module stays free of heavy imports.

    `capability` imports json and re and nothing else, but it is not on the import-free list this
    file's head pins, so it is imported inside the function rather than at module scope.
    """
    from .capability import TASK_CHOICES
    return TASK_CHOICES


#: The words that select a mode rather than a subcommand. `split_mode` peels them off, so argparse
#: never sees them and they cannot be given a subparser with a help page of its own.
MODES = ("abliterate", "kageyoshi", "auto")

#: The flags the kageyoshi preset resolves from the model, and therefore the ones it stands down
#: from when the caller sets them by hand.
#:
#: DUPLICATED FROM `cli._KAGEYOSHI_BUDGET_FLAGS` because this module may not import `cli`: that
#: costs torch and 2.8 seconds, which is the whole reason this file exists. So the list is pinned
#: by a test instead of by an import, the same way the flag counts in the help text are counted
#: rather than typed.
KAGEYOSHI_PRESET_FLAGS = (
    "--trials", "--patience", "--max-directions", "--kl-scale", "--search", "--per-component",
    "--mlp-off", "--dir-prompts", "--eval-refusal", "--eval-kl", "--eval-refusal-final",
    "--top-rescore",
)

#: Every line in the two pages below stays inside 79 columns, so an 80-column terminal does not
#: re-wrap them into something the author never laid out.
_ABLITERATE_HELP = """\
usage: senbonzakura abliterate MODEL [flags]

Abliterate at the flat defaults, for a run you are driving yourself.

Naming `abliterate` opts out of the auto-scaled preset that `senbonzakura
MODEL` and `senbonzakura kageyoshi MODEL` both use. Nothing is read off the
model: every flag stays at its documented default until you set it.

Use it to hold one setting still across several models, or to reproduce a run
whose flags you already have. For a first run, use `kageyoshi` instead.

example:
  senbonzakura abliterate Qwen/Qwen2.5-0.5B-Instruct \\
      --track default --out edited \\
      --trials 120 --min-directions 2 --max-directions 2

every flag:  senbonzakura --help-all
"""

_KAGEYOSHI_HELP = """\
usage: senbonzakura {word} MODEL [flags]

Abliterate with the auto-scaled preset. This is the recommended way to run it.

It loads the model once, reads the architecture (dense, fused MoE or expert
list) and the parameter count, then sizes the search and turns on the quality
levers from what it found. You choose the model and where the output goes.

Anything you set yourself wins: the preset keeps your value, skips its own, and
says which in the log. These are the flags it would otherwise choose for you.

{preset}

example:
  senbonzakura {word} Qwen/Qwen2.5-0.5B-Instruct \\
      --track default --out edited

`senbonzakura Qwen/Qwen2.5-0.5B-Instruct`, with no mode word, does the
same thing.
`auto` and `kageyoshi` are the same mode under two names.

every flag:  senbonzakura --help-all
"""


def mode_help(word):
    """The page a named mode prints for `--help`.

    WHAT THIS FIXES. `abliterate --help`, `kageyoshi --help` and `auto --help` printed the
    top-level page byte for byte, identical to `senbonzakura --help`: a usage line reading
    `senbonzakura [-h] ...`, no mention of the word that was typed, and the whole command list
    underneath. So asking what `kageyoshi` does answered with a page that names it once, in a list
    of everything else.

    A page rather than a subparser, because these words are modes: `split_mode` removes them before
    argparse runs, and the flag surface behind all three is the same parser.
    """
    if word == "abliterate":
        return _ABLITERATE_HELP
    import textwrap

    return _KAGEYOSHI_HELP.format(
        word=word,
        # `break_on_hyphens=False` or the filler splits `--top-rescore` across two lines and
        # prints a flag nobody can type.
        preset=textwrap.fill("  ".join(KAGEYOSHI_PRESET_FLAGS), width=79,
                             break_on_hyphens=False,
                             initial_indent="  ", subsequent_indent="  "))


def split_mode(argv):
    """Peel off the mode word, returning (bankai, remaining argv).

    IT ALSO ANSWERS `--help` FOR THE MODE WORD, and does so here rather than in `entry` because
    there are two doors into the abliterator: the console script through `entry.main`, and `python
    -m senbonzakura.cli` through `cli.main`. Both call this function and nothing else in common
    before the parse. Two doors into one command disagreeing about what it prints is this project's
    most-repeated defect shape, so the answer lives in the one place both of them pass through.

    `kageyoshi` is a real subcommand rather than an argv[0] trick: it runs the abliterator with
    the auto-scaled best-effort preset, resolved after the model loads once the architecture and
    parameter count are known. `auto` is a plain-English alias for it, not a second mode; someone
    meeting this tool for the first time should not have to know a Japanese sword release to get
    the setting that thinks for them.

    THE BARE FORM IS THE PRESET TOO, since 2026-09-27. `senbonzakura <model>` and
    `senbonzakura kageyoshi <model>` now do the same thing, because the shortest command anybody
    types should give the best result the tool can produce. It used to run the flat defaults, so the
    shortest command was the worst one and nothing about either command line said so.

    `abliterate`, named out loud, is the opt out: the flat defaults, for a run somebody is driving
    by hand. That leaves a route to full manual control without making the short form the booby
    trap.

    Here rather than in `cli` because the entry point has to know which words are modes before it
    can parse, and it must reach that answer without importing torch.
    """
    if argv and argv[0] in MODES and {"-h", "--help"} & set(argv[1:]):
        # Not `--help-all`: that is a real action on the flag parser and still prints every flag.
        print(mode_help(argv[0]))
        raise SystemExit(0)
    if argv and argv[0] in ("kageyoshi", "auto"):
        return True, argv[1:]
    # `abliterate` NAMED EXPLICITLY IS THE OPT OUT, and the bare form is not.
    #
    # Operator decision 2026-09-27: `senbonzakura <model>` should give the best result the tool can
    # produce, because that is what somebody typing the shortest thing means. It used to run the
    # flat defaults, so the shortest command was the worst one, and the difference was invisible:
    # two commands that look like the same request produced different searches and nothing said so.
    #
    # Saying `abliterate` out loud still means "the defaults, and I will set what I want myself",
    # which keeps a route to a fully hand-driven run. Anything set by hand wins in either mode, so
    # the preset is no longer a reason to avoid the short form.
    if argv and argv[0] == "abliterate":
        return False, argv[1:]
    return True, list(argv)


def loader_parser(*, model_help="HF model id or local path", four_bit_help=None,
                  chat_template=True, model_required=True):
    """The flags every entry point needs in order to LOAD a model, defined once.

    A parent parser rather than a copy in each of the four commands. The copies had already
    drifted in ways that matter: `--device` carried help text in three places and none in the
    fourth, `--load-in-4bit` existed on two of the four forward-only paths, and each `--model`
    described itself differently. Worse, the same drift in the *prompt* rendering beside these
    flags is what put the compass's read-out at the wrong position (see `render_chat`), so
    "four near-copies of the loading surface" is not a tidiness complaint.

    `four_bit_help` lets the abliterator say that it REJECTS the flag while still accepting it,
    which is what turns an obscure failure at bake time into a sentence at startup.
    """
    ap = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    # `model_required=False` ONLY for the abliterate parser, which also takes the model as a
    # positional so that `senbonzakura Qwen/Qwen2.5-0.5B-Instruct` works. It still refuses a run with no
    # model at all; the check just moves from argparse to `resolve_model`, where it can say which
    # of the two ways to give it you meant to use. Every other command keeps the flag required.
    ap.add_argument("--model", required=model_required, default=None, help=model_help)
    ap.add_argument("--device", default="cuda",
                    help="cuda, cuda:N, or cpu (default: cuda). A machine with no card needs "
                         "--device cpu; `senbonzakura doctor` says what this one has.")
    ap.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true",
                    help="allow models that ship custom modelling code (some Hub models need "
                         "it); off by default.")
    # Omitted for a command whose measurement does not depend on prompt format, so it does not
    # offer a knob that would change nothing.
    if chat_template:
        ap.add_argument("--chat-template", dest="chat_template", default="",
                        help="a Jinja chat template for a model that ships none: a path to one, "
                             "or a bundled name ('plain'). Prompt format decides every "
                             "measurement here, so the run records which template it used.")
    ap.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true",
                    help=four_bit_help or ("load in 4-bit (bitsandbytes nf4) to measure a large "
                                           "model on low VRAM. Safe on the forward-only paths; "
                                           "the abliterator refuses it, because the weight bake "
                                           "rewrites tensors and needs full precision."))
    return ap


#: The flags a first run actually needs, by `dest`. Everything else stays real, stays accepted and
#: stays documented; it just does not greet somebody who typed `--help` to find out what this is.
#:
#: WHY THIS EXISTS. The default command carries 69 flags and its help is 470 lines. That surface is
#: honest for the research side of this tool, where the knobs ARE the product, and it is a wall in
#: front of the other side, where somebody wants a model at the end. Chosen by counting: these are
#: the flags the user-facing pages actually tell a reader to type, plus the three that only matter
#: when something is wrong (`--resume`, `--hf-token`, `--trust-remote-code`).
CORE_FLAGS = frozenset({
    "model_positional", "model", "track", "out", "device", "method", "trials", "max_directions",
    "load_in_4bit", "hf_token", "trust_remote_code", "resume", "seed", "help", "help_all",
    "version",
})


#: Flags that are not in `CORE_FLAGS` and still belong under `options:` beside the model: the rest
#: of the loading surface, and the completion script when shtab is installed.
_STAYS_IN_OPTIONS = frozenset({"chat_template", "print_completion"})

#: Which heading each remaining flag sits under, by `dest`, and the order the headings print in.
#:
#: WHY THIS EXISTS. `--help-all` ran 417 lines of flags under one `options:` heading, with nothing
#: between the VRAM throttle and the capability probe to say a reader had moved from one subject to
#: another. The footer on the short page already promised groups that existed nowhere, so the page
#: described a structure it did not have.
#:
#: Keyed by `dest` rather than by flag, so a pair like `--matched-scoring` and
#: `--no-matched-scoring` cannot land under two different headings.
FLAG_GROUPS = {
    # NO CORE FLAG IS FILED HERE, deliberately. A core flag stays visible on the short page, so
    # filing one under a heading put a `the search:` section holding a single flag at the bottom of
    # `--help`, which reads as a page that lost the rest of its content. Groups therefore cover
    # exactly the flags the short page hides, which is also what its footer claims.
    "the search": (
        "patience", "search", "kl_scale", "max_kl", "layer_lo", "layer_hi",
        "warm_start", "top_rescore", "eval_refusal_final", "study_db", "no_persist_study",
        "bake_config", "bench_only",
    ),
    "the directions": (
        "min_directions", "direction_clusters", "separation_statistic", "matched_scoring",
        "harmless_matched", "hedge_ds", "clean_ds", "no_good_orth", "no_norm_restore",
        "skip_conv_ablation", "per_component", "mlp_off", "sparsity", "ablation_rounds",
    ),
    "the prompts and the scoring": (
        "dir_prompts", "good_ds", "text_column", "eval_refusal", "eval_kl", "gen_tokens",
        "short_budget_ok", "low_refusal_ok", "inspect", "inspect_n",
    ),
    "the capability probe": (
        "capability_eval", "capability_n", "capability_task", "capability_max_new",
        "slow_probe_ok",
    ),
    "the machine": (
        "gen_batch", "gpu_min_free_frac", "max_pause_s", "no_throttle", "background_mode",
        "external_pressure_mb", "attn_impl",
    ),
    "the output": (
        "base_licence", "base_licence_link", "free_base_model", "json_events", "no_panel",
    ),
}


def _file_flags_under_headings(ap):
    """Move every flag `FLAG_GROUPS` names out of `options:` and under its heading.

    A move after the fact rather than an argument group per flag at declaration time, for the same
    reason the suppression below happens at the end: one table that can be read against the parser
    is checkable, and fifty scattered group names are not.
    `tests/test_the_full_help_has_headings.py` asserts that every flag is either core or filed.

    A flag nobody has filed stays where it is. The heading is presentation, and a missing entry
    should show up as a test failure rather than as a command that will not start.
    """
    where = {dest: title for title, dests in FLAG_GROUPS.items() for dest in dests}
    # `_action_groups[1]` is argparse's own optionals group, which is where every flag declared on
    # `ap` and every flag inherited from `loader_parser` has landed.
    options = ap._action_groups[1]                       # noqa: SLF001 - no public accessor
    groups = {title: ap.add_argument_group(title) for title in FLAG_GROUPS}
    for action in list(options._group_actions):          # noqa: SLF001
        title = where.get(action.dest)
        if title is None:
            continue
        options._group_actions.remove(action)           # noqa: SLF001
        groups[title]._group_actions.append(action)     # noqa: SLF001


class _HelpAll(argparse.Action):
    """Print the full help, which is what `--help` printed before the split.

    A separate parser rather than a stored one: this parser has already had its help text
    suppressed by the time anybody can type the flag, and un-suppressing in place would mean
    holding two descriptions of every flag.
    """

    def __init__(self, option_strings, dest, **kw):
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kw)

    def __call__(self, parser, _namespace, _values, _option_string=None):
        build_parser(full=True).print_help()
        parser.exit()


def build_parser(full=False):
    """The default command's parser. `full=True` is the long help, behind `--help-all`.

    Suppression happens at the END, after every flag is declared, so a flag cannot be added in a
    way that quietly escapes it. The parse is identical either way: this changes what is printed,
    never what is accepted, and `tools/research/audit_flags.py` and the documentation guards read
    the full form for exactly that reason.
    """
    ap = argparse.ArgumentParser(
        allow_abbrev=False,
        prog="senbonzakura",
        # WRAPPED BY HAND, because RawDescriptionHelpFormatter does not wrap. This was one
        # 213-column line on the first page anybody sees.
        description=("Refusal abliteration for transformer language models, with the\n"
                     "instruments to measure what the edit cost. A quality-guarded Optuna\n"
                     "(NSGA-II) search over windowed, per-component, multi-directional\n"
                     "weight ablations."),
        epilog=(
            "commands:\n"
            "  abliterate       edit a model at the flat defaults, for a run you are\n"
            "                   driving by hand. Naming it opts out of the auto-scaled\n"
            "                   preset.\n"
            "  kageyoshi        edit a model with the auto-scaled preset: it reads the\n"
            "                   architecture and size, sizes the search and turns on every\n"
            "                   quality lever. Anything you set by hand wins.\n"
            "  measure          every instrument against one model, into one table:\n"
            "                   refusal, harm recognition, fluency, capability, and the\n"
            "                   coherence cost when --baseline is given.\n"
            "  compass          harm discrimination as the harmful/benign logit margin\n"
            "                   (AUC), with its construct-validity controls beside it.\n"
            "  score            refusal, hedging, the Heretic keyword rate and broken\n"
            "                   output, on a fixed eval set.\n"
            "  coherence        perplexity of a fixed neutral passage: the coherence cost.\n"
            "  drift            one coherence ruler applied to any model after the fact, so\n"
            "                   models edited by different tools land on the same scale.\n"
            "  track            build an evaluation track with a checked fit / search /\n"
            "                   measure split.\n"
            "  corpora          fetch and pack the refusal corpora. A released wheel\n"
            "                   already carries them; a build from the repository does not.\n"
            "  auto             alias for kageyoshi, for anyone who has not met the name.\n"
            "  harm-recognition alias for compass, for the same reason.\n"
            "  interactive      a guided walk through the choices that decide whether a run\n"
            "                   means anything. It prints each command before running it.\n"
            "  validate         ask whether a direction set carries refusal or carries\n"
            "                   topic: leave-one-cluster-out generalisation, a random-\n"
            "                   direction floor, and a sweep of direction count compared at\n"
            "                   matched refusal removal.\n"
            "  capability       what the edit cost, on a task the model either gets right\n"
            "                   or does not. Refusal rates and KL cannot see reasoning\n"
            "                   loss.\n"
            "  judge            check a grading model against reference labels before\n"
            "                   letting it grade anything. Reports agreement above chance,\n"
            "                   and exits non-zero when not certified.\n"
            "  report           assemble a run's artefacts into the card that goes beside\n"
            "                   the weights. States only what the artefacts support.\n"
            "  head-to-head     run this tool and others over the same model, corpus and\n"
            "                   budget on one machine, then report what separates them.\n"
            "  convert          turn edited weights into a GGUF with the pinned converter,\n"
            "                   optionally quantising in the same command.\n"
            "  quantise         shrink a GGUF with the pinned llama-quantize, then read it\n"
            "                   back to confirm the quantisation asked for.\n"
            "  fetch            download a model file and prove it is the one asked for:\n"
            "                   length, GGUF header, architecture and quantisation.\n"
            "  imatrix          compute an importance matrix so a quantisation keeps the\n"
            "                   weights that matter, and record what it was calibrated on.\n"
            "  check            read a result file, from this tool or another, and report\n"
            "                   how the number could be wrong. Needs no model, corpus or\n"
            "                   card.\n"
            "  baseline         record one measurement as the baseline the gate compares\n"
            "                   against, with the conditions it was measured under. Needs\n"
            "                   no model or card.\n"
            "  gate             compare a measurement against a recorded baseline and fail\n"
            "                   a build on a regression. Refuses when the two are not\n"
            "                   comparable.\n"
            "  prereg           check a pre-registration, and whether a run did what it\n"
            "                   promised. Needs no model, corpus or card.\n"
            "  setup            put the right build of torch on this machine: pip picks by\n"
            "                   platform, not by hardware.\n"
            "  doctor           check this install can do the job: pins, the binary, every\n"
            "                   architecture module, the bundled data.\n"
            "each command takes --help of its own, e.g. `senbonzakura compass --help`"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[loader_parser(
            model_help="HF model id or local path to abliterate",
            four_bit_help="NOT supported by the abliterator: the weight bake needs full "
                          "precision. Use it with the scorer "
                          "(python -m senbonzakura.score --load-in-4bit) to measure a model "
                          "on low VRAM.",
            model_required=False)])
    # THE ONE-COMMAND FORM: `senbonzakura Qwen/Qwen2.5-0.5B-Instruct`, with everything else defaulted.
    #
    # `--model` still works and is what every run spec, every README example and every holst job
    # already passes, so nothing that exists breaks. This is an additional way to say the same
    # thing, for the case where somebody has just installed the tool and wants to see it do
    # something. `entry.main` already routes anything that is not a subcommand here, so the only
    # missing piece was a parser that would accept it.
    #
    # `nargs="?"` rather than required, because `--model` has to keep working on its own, and
    # `metavar` so the usage line reads MODEL rather than the dest name.
    ap.add_argument("model_positional", nargs="?", default=None, metavar="MODEL",
                    help="the model to abliterate, given without a flag. `senbonzakura "
                         "Qwen/Qwen2.5-0.5B-Instruct` is the whole command: the track, the output directory "
                         "and the search are all defaulted. Equivalent to --model.")
    try:   # optional shell completion; degrade gracefully if shtab is not installed
        import shtab
        shtab.add_argument_to(ap, ["--print-completion"],
                              help="print a bash/zsh/tcsh shell completion script and exit")
    except ImportError:
        pass
    ap.add_argument("--out", default="abliterated", help="directory to write the abliterated model to")
    ap.add_argument("--dir-prompts", type=int, default=256, help="contrast prompts per side for direction extraction")
    ap.add_argument("--eval-refusal", type=int, default=64, help="bad-eval prompts for the refusal score")
    ap.add_argument("--eval-kl", type=int, default=64, help="harmless prompts for the KL score")
    ap.add_argument("--trials", type=int, default=60,
                    help="how many search trials to run (default: 60). The search is NSGA-II over "
                         "refusal against quality, so more trials buy a better-explored frontier "
                         "rather than a better single answer; see --patience to stop early when "
                         "it has stopped improving")
    ap.add_argument("--kl-scale", type=float, default=4.0,
                    help="weight on KL in the SCALAR objective (higher = protect quality more). It "
                         "does nothing under the default --search pareto, which carries KL as its "
                         "own axis, and nothing under --search scalar while KL stays below the "
                         "ceiling. Setting it and seeing no change is expected, not a fault")
    ap.add_argument("--layer-lo", type=float, default=0.3, help="search layers from this fraction of depth")
    ap.add_argument("--layer-hi", type=float, default=0.8,
                    help="search layers up to this fraction of depth (default: 0.8). The window is "
                         "a fraction rather than a layer number so the same setting means the same "
                         "thing on models of different depths")
    ap.add_argument("--gen-tokens", type=int, default=_DEFAULT_BUDGET,
                    help=f"how many tokens each reply gets while the search scores it (default: "
                         f"{_DEFAULT_BUDGET}). THE SETTING MOST LIKELY TO MAKE A REFUSAL RATE READ "
                         f"LOW: a refusal the model never reaches is not counted, and the search "
                         f"then picks configurations whose refusal lands after the cutoff. Find "
                         f"the budget your model needs with `senbonzakura score --length-sweep`. "
                         f"Below {_VISIBILITY_FLOOR} the run refuses without --short-budget-ok")
    ap.add_argument("--short-budget-ok", dest="short_budget_ok", action="store_true",
                    help=f"allow --gen-tokens below {_VISIBILITY_FLOOR}, which is otherwise "
                         f"refused. For a smoke test, not for a number you will quote: at that "
                         f"budget the result is about the budget and not about the model")
    ap.add_argument("--gen-batch", type=int, default=16, dest="gen_batch",
                    help="max prompts per generation batch (the ceiling the adaptive VRAM throttle "
                         "ramps up to; it shrinks below this automatically when the card is busy).")
    ap.add_argument("--gpu-min-free-frac", type=float, default=0.06, dest="gpu_min_free_frac",
                    help="pause generation while USABLE VRAM (free plus senbon's own reclaimable "
                         "cache) is below this fraction of the card. Because it counts senbon's own "
                         "cache, a model that simply fills the card does not pause; only another app "
                         "(a game, a browser) taking the GPU triggers a pause, with resume on free-up.")
    ap.add_argument("--max-pause", type=float, default=None, dest="max_pause_s",
                    help="safety cap (seconds) on how long to wait for VRAM headroom before pushing on "
                         "regardless; default waits indefinitely so a busy card never crashes the run.")
    ap.add_argument("--no-throttle", action="store_true", dest="no_throttle",
                    help="disable the adaptive VRAM throttle (fixed batch, no pause/resume). Use only "
                         "when senbon has the card to itself and you want maximum, unpaced throughput.")
    ap.add_argument("--background", action="store_true", dest="background_mode",
                    help="good-gaming-citizen mode: run in the background and YIELD the GPU (pause "
                         "generation) whenever a foreground app (a game) is on the card, resuming when "
                         "it closes. Frees compute, not just VRAM, so the game stays smooth.")
    ap.add_argument("--external-pressure-mb", type=int, default=500, dest="external_pressure_mb",
                    help="in --background mode, how much VRAM a non-senbon process must hold to count "
                         "as a foreground app worth yielding to (default 500 MB).")
    ap.add_argument("--bench-only", action="store_true", help="load, extract, run 1 default-strength "
                                                              "ablation + print refusals, no search")
    # `default` was findable only by omitting --track and reading the refusal, and the refusal
    # buried it under three paragraphs of Hub-package advice. It is the easiest way to get a first
    # run working, so it belongs in the one place a user looks first.
    ap.add_argument("--track", default=TRACK_AUTO,
                    help="a directory holding bad_ds / good_ds / bad_eval_ds, as built by "
                         "`senbonzakura track`. Pass 'default' for the bundled evaluation track, "
                         "which needs no network and is the quickest way to a first run. Left out, "
                         "it takes ./track when that exists and the bundled one otherwise, and says "
                         "in the log which it chose.")
    ap.add_argument("--good-ds", default=None, help="override the harmless dataset dir (for a matched-form contrast)")
    # Every dataset argument above accepts a save_to_disk directory, a .txt/.csv/.json/.jsonl/
    # .parquet file, or a Hub id, optionally with `::split[:N]`. These two are the knobs the
    # detection cannot work out on its own.
    ap.add_argument("--text-column", dest="text_column", default=None,
                    help="column holding the prompt, when it is not one of the names this "
                         "detects (text, prompt, instruction, goal, behavior, question, ...)")
    ap.add_argument("--hf-token", dest="hf_token", default=None,
                    help="token for a gated or private Hub dataset; defaults to $HF_TOKEN. "
                         "Prefer the environment variable: an argument is visible in `ps`.")
    ap.add_argument("--attn-impl", dest="attn_impl", default=None,
                    help="attention implementation to request (eager / sdpa / flash_attention_2); "
                         "default lets transformers choose (sdpa).")
    ap.add_argument("--inspect", nargs=2, type=float, default=None, metavar=("LAYER", "STRENGTH"),
                    help="print real harmful+harmless generations at (layer, strength), pre and post "
                         "ablation, then exit")
    ap.add_argument("--inspect-n", type=int, default=8, help="prompts per side to print in --inspect")
    ap.add_argument("--max-directions", type=int, default=3,
                    help="CEILING on refusal directions per layer, not a count. Each trial draws "
                         "its own number between --min-directions and this, and records what it "
                         "chose as `num_directions`. Set both to the same number to pin it, which "
                         "is what turns a ceiling into an experiment")
    ap.add_argument("--direction-clusters", type=int, default=8,
                    help="how many refusal modes to look for per layer. Each cluster of harmful "
                         "prompts proposes one candidate direction, and the best separators are "
                         "kept up to --max-directions. Independent of that ceiling, so the "
                         "candidate set does not move when the budget does.")
    # `metavar` so argparse's generated usage line does not carry the four choices inline. It was
    # 86 columns on an 80-column terminal, which wraps into something unreadable on the one surface
    # a reader meets before anything else. The choices are named in the help text instead, where
    # there is room to say what each one means. Same treatment as `--method`.
    ap.add_argument("--separation-statistic", dest="separation_statistic",
                    choices=_separation.CHOICES, default=_separation.DEFAULT_STATISTIC,
                    metavar="STATISTIC",
                    help="which statistic decides whether a candidate axis carries refusal rather "
                         "than topic. One of: " + ", ".join(_separation.CHOICES) + ". 'cohens-d' "
                         "(default) has a threshold that means a different thing at every cluster "
                         "size; 'variance-ratio' is the ANOVA F, whose null is 1.0 at any size.")
    ap.add_argument("--matched-scoring", dest="matched_scoring", action="store_true", default=True,
                    help="judge each candidate direction against the harmless prompts nearest it "
                         "in content rather than against the harmless set at large, so that "
                         "refusal is the only thing left varying and not the subject matter. On by "
                         "default, and the only setting that separates the two.")
    ap.add_argument("--no-matched-scoring", dest="matched_scoring", action="store_false",
                    help="score candidates against the harmless set at large. For reproducing an "
                         "older run and nothing else: that comparison cannot tell refusal from "
                         "subject matter.")
    ap.add_argument("--capability-eval", dest="capability_eval", default="bundled",
                    help="what the capability probe measures against. 'bundled' (default) is 256 "
                         "grade-school arithmetic questions that ship with the package, so it "
                         "works offline. Give a graded benchmark (a question column and an answer "
                         "column, e.g. openai/gsm8k:main::test) for your own, or an empty string "
                         "to turn the probe off. No other metric here can see reasoning loss.")
    ap.add_argument("--capability-n", dest="capability_n", type=int, default=200,
                    help="how many items the capability probe uses (default: 200, 0 = off). Before "
                         "and after are scored on the same items, so the comparison is paired. "
                         "Much below 200 the error bar covers the effect this edit is expected to "
                         "have, which reads like a measurement and is not one.")
    # Same reason as `--separation-statistic` above: five choices inline made this 93 columns.
    ap.add_argument("--capability-task", dest="capability_task",
                    choices=_capability_tasks(), default="numeric", metavar="TASK",
                    help="how the probe grades. One of: " + ", ".join(_capability_tasks())
                         + " (default: numeric). `senbonzakura capability --help` says what each "
                           "one measures.")
    ap.add_argument("--capability-max-new", dest="capability_max_new", type=int, default=512,
                    help="token budget per probe answer (default: 512). A worked solution is long, "
                         "and a budget that cuts it off measures the budget rather than the model. "
                         "A truncated answer is counted as ungradeable, never as wrong.")
    ap.add_argument("--low-refusal-ok", dest="low_refusal_ok", action="store_true",
                    help="edit a model that hardly refuses anything to begin with. The run measures "
                         # `%%` because argparse runs this through percent formatting, and a bare
                         # `%` raises "badly formed help string" from `add_argument`, which breaks
                         # every command rather than only this flag.
                         "the baseline refusal rate about a minute in, and below 5%% it stops: "
                         "there is too little refusal to find a direction for, so the search "
                         "would spend its whole budget, pay the coherence cost and hand back a "
                         "model refusing about as often as it started.")
    ap.add_argument("--slow-probe-ok", dest="slow_probe_ok", action="store_true",
                    help="run the capability probe on a CPU even when it will take hours. The probe "
                         "defaults are sized for a GPU, where they are minutes; on a CPU with a "
                         "1.7B model they are about four hours, so the run stops and says so "
                         "rather than looking like a hung one all afternoon.")
    ap.add_argument("--ablation-rounds", dest="ablation_rounds", type=int, default=0,
                    help="how many times to alternate restoring the row lengths and removing the "
                         "direction again (default: 0, a single pass). A single pass leaves 5%% to "
                         "46%% of the direction behind, because restoring the lengths undoes part "
                         "of the cut; 4 rounds removes it and keeps the lengths. Off by default "
                         "because turning it on moves every number a run produces.")
    # `metavar` so the four choices do not put a 92-column line in the usage block. They move into
    # the help, where all four are now described: `searched-one-direction` was a choice the parser
    # accepted and the help never mentioned, so a `metavar` alone would have made it undiscoverable.
    # It is the cell that isolates the search from the direction count, which is this project's own
    # central comparison, so leaving it unnamed was the worse of the two faults.
    ap.add_argument("--method", choices=_methods.CHOICES, default=_methods.DEFAULT_METHOD,
                    metavar="RECIPE",
                    help="which ablation recipe to run, recorded in abliteration.json. "
                         "'searched' (default) optimises how much to ablate, where, and over as "
                         "many directions as separate. 'searched-one-direction' searches the same "
                         "way but removes only the global difference of means, which isolates the "
                         "search from the direction count. 'single-pass' fixes full strength on one "
                         "direction with no search. 'single-pass-raw' is the same without restoring "
                         "the original row norms, as a control.")
    ap.add_argument("--free-base-model", dest="free_base_model", action="store_true",
                    help="delete the local base model directory just before writing the output, "
                         "when there is not room for both. Irreversible, and off by default. "
                         "Refused when the source is a shared Hugging Face cache, when --out is "
                         "that directory or sits inside it, and when there is already room.")
    ap.add_argument("--harmless-matched", dest="harmless_matched", default="",
                    help="a second harmless set on the SAME subjects as the harmful one, used as "
                         "the pool --matched-scoring draws its controls from. Without it the "
                         "controls are the nearest rows of the ordinary harmless set. Requires "
                         "--matched-scoring. How well it matched is reported as matching_quality "
                         "in abliteration.json, where a value near 1.0 means it did not.")
    # DECISION Q-37. Without this the run saves weights and writes no card, because `modelcard`
    # refuses to infer the base model's licence: a model's terms are not derivable from its
    # weights and a wrong guess is worse than a blank one. Asking here is asking at the moment
    # the operator has just chosen a base model. Given it, the run writes a card that could
    # actually be published; not given it, the run writes none and says why, rather than
    # manufacturing one marked UNRESOLVED that somebody eventually publishes.
    ap.add_argument("--base-licence", dest="base_licence", default="",
                    help="the BASE model's licence (apache-2.0, mit, gemma, llama3.2, other). "
                         "Given this, the run writes a README.md model card beside the saved "
                         "weights; without it no card is written, because this cannot be "
                         "inferred from the model and a wrong guess is worse than a blank.")
    ap.add_argument("--base-licence-link", dest="base_licence_link", default="",
                    help="URL for the base model's licence text, where one exists. It travels "
                         "into the card so a reader can go and read the terms they are bound by.")
    ap.add_argument("--sparsity", type=float, default=0.0,
                    help="sparse surgery: the fraction of output rows to LEAVE untouched per "
                         "weight, editing only the most refusal-writing ones. 0.0 (default) edits "
                         "every row; 0.3 leaves the quietest 30%% alone, for less collateral. "
                         "--ablation-rounds honours the same mask.")
    ap.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=True,
                    help="seed the search with one sane difference-of-means configuration so it "
                         "starts from a known-decent point rather than cold random sampling. On by "
                         "default; --no-warm-start runs the cold search.")
    ap.add_argument("--no-good-orth", action="store_true", dest="no_good_orth",
                    help="do NOT orthogonalise the refusal direction against the harmless mean; use "
                         "the raw difference of means instead. Turns off the projection grimjim "
                         "calls 'projected abliteration', which is on by default. A control arm.")
    ap.add_argument("--no-norm-restore", dest="no_norm_restore", action="store_true",
                    help="CONTROL ARM ONLY, and it makes a worse model on purpose. Remove the "
                         "directions without putting each weight row's original length back. That "
                         "restore is what keeps an edited model coherent. Not the same as "
                         "--no-good-orth, which changes how the directions are found.")
    ap.add_argument("--skip-conv-ablation", dest="skip_conv_ablation", action="store_true",
                    help="CONTROL ARM ONLY, and it makes a PARTIAL abliteration by construction. "
                         "Leave the output projections of non-attention sequence mixers untouched "
                         "on a hybrid architecture, which on some models is most of the stack. "
                         "Answers whether refusal travels that path, by comparison against a run "
                         "without it. Every skipped layer is warned about and recorded.")
    from . import events as _events
    _events.add_argument(ap)
    # Imported here beside the events flag, for the same reason: both modules define their own flag
    # next to the behaviour it controls, and neither is heavy (`panel` imports rich lazily, so
    # `--help` does not pay for it).
    from . import livedisplay as _livedisplay
    _livedisplay.add_argument(ap)
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for the Optuna sampler (default 42). Vary it to measure run-to-run "
                         "spread: a single run tells you nothing about whether a gap between two "
                         "configurations is real. Note GPU kernels are not bit-deterministic, so a "
                         "fixed seed reproduces the search path, not the last decimal of a score.")
    ap.add_argument("--search", choices=["pareto", "scalar"], default="pareto",
                    help="pareto (default): NSGA-II maps the whole refusal-against-KL frontier and "
                         "the knee is picked, the most uncensored point that is still intact. "
                         "scalar: one weighted objective, searched with TPE.")
    ap.add_argument("--per-component", dest="per_component", action="store_true", default=True,
                    help="tune attn.o_proj and mlp.down_proj SEPARATELY (Heretic-style). The MLP "
                         "profile may go to zero (leave the MLP untouched), which often preserves "
                         "intelligence. This is the default.")
    ap.add_argument("--uniform", dest="per_component", action="store_false",
                    help="apply ONE strength profile to both components instead of tuning them "
                         "separately.")
    ap.add_argument("--mlp-off", dest="mlp_off", action="store_true",
                    help="pin mlp.down_proj ablation to zero (attention-only). Tests the "
                         "'attention carries refusal, MLP carries capability' hypothesis and "
                         "removes the d-profile dimensions from the search entirely.")
    ap.add_argument("--hedge-ds", default=None,
                    help="dir of a HEDGED-compliance dataset (moralising-but-complying answers). "
                         "When given, a hedged-vs-clean contrast direction is folded into the "
                         "ablated basis, so the search can remove the disclaimer/hedging axis that "
                         "the difference-of-means (hard-refusal) direction misses.")
    ap.add_argument("--clean-ds", default=None,
                    help="dir of CLEAN (disclaimer-free) compliance for the hedged contrast; "
                         "defaults to --good-ds / <track>/good_ds.")
    ap.add_argument("--min-directions", dest="min_directions", type=int, default=1,
                    help="the FEWEST directions a trial may use. --max-directions is only a "
                         "ceiling, so 'up to two' is not 'two': set both to the same number to pin "
                         "the budget, which is what turns a K comparison into an experiment rather "
                         "than a mixture.")
    ap.add_argument("--max-kl", dest="max_kl", type=float, default=None,
                    help="the most coherence drift you will accept, as KL (default: filter at "
                         f"{KL_CEIL}, surcharge above {KL_TARGET}). The search then returns the "
                         "biggest refusal reduction it can manage under this figure, and refuses "
                         "rather than returning a configuration that misses it. This is the flag "
                         "that puts this tool and Heretic at one operating point.")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop the search early if no trial improves the best scalarised score for "
                         "this many consecutive trials (0 = run all --trials).")
    ap.add_argument("--eval-refusal-final", type=int, default=128,
                    help="re-score the top frontier candidates on this many held-out bad-eval "
                         "prompts before picking the knee (default: 128, 0 = off). Without it the "
                         "winner is the best of N draws over the same small set every trial was "
                         "scored on, so its refusal figure describes the rows it was selected on. "
                         "Costs one scoring pass, against a search measured in hours")
    ap.add_argument("--top-rescore", type=int, default=6,
                    help="how many frontier candidates to re-score with --eval-refusal-final.")
    ap.add_argument("--study-db", default=None,
                    help="persist the Optuna study to this SQLite file (default: <out>/senbon-study.db, "
                         "so a killed run resumes with --resume instead of re-searching). An "
                         "existing study at the older <track>/senbon-study.db is still picked up.")
    ap.add_argument("--no-persist-study", action="store_true", dest="no_persist_study",
                    help="do NOT persist the Optuna study (in-memory only). A crash then loses the "
                         "search; the persistent default is the safer choice for a long paid run.")
    ap.add_argument("--resume", action="store_true",
                    help="resume a persisted study; continues where an interrupted search left off, "
                         "and if the study already finished, skips straight to bake+save.")
    ap.add_argument("--bake-config", default=None, dest="bake_config",
                    help="skip the search entirely: load a saved best-config.json and bake+save that "
                         "config directly. Recovers a crashed save in minutes instead of re-searching.")
    ap.add_argument("--version", action="version", version=f"senbonzakura {__version__}")
    ap.add_argument("--help-all", action=_HelpAll, dest="help_all",
                    # Filled in below, once every flag has been declared and can be counted. It
                    # used to say "there are 69 in total" as a literal, while the numbers in the
                    # epilog beside it were computed, so the two drifted apart: by 2026-09-27 the
                    # computed pair read 16 and 56 against a typed 69. A reader who adds them up
                    # finds the disagreement, which is how a first-time reader found this one.
                    help=argparse.SUPPRESS)

    # COUNTED, NOT TYPED, and counted over FLAGS rather than over argparse actions.
    #
    # `len(ap._actions)` includes the model positional, which is not a flag, so the old arithmetic
    # reported 16 flags where 15 flags and one positional were shown. Two numbers were wrong in the
    # same sentence: one because it was stale and one because it counted the wrong things.
    flags = [a for a in ap._actions if a.option_strings]   # noqa: SLF001 - no public accessor
    core = sum(1 for a in flags if a.dest in CORE_FLAGS)
    for action in flags:
        if action.dest == "help_all":
            action.help = (
                "show every flag, grouped by what it controls. This page shows the ones a run "
                f"needs; there are {len(flags)} in total.")

    _file_flags_under_headings(ap)

    if not full:
        # AT THE END, ON WHAT WAS ACTUALLY DECLARED, so a flag added later cannot escape the
        # split by being added somewhere this function does not look.
        hidden = 0
        for action in ap._actions:   # noqa: SLF001 - argparse exposes no public accessor
            if action.dest not in CORE_FLAGS and action.help is not argparse.SUPPRESS:
                action.help = argparse.SUPPRESS
                hidden += 1
        # Wrapped for the same reason as the description above.
        ap.epilog += (
            f"\n\nThis page shows the {core} flags a run needs, and the model it edits.\n"
            f"{hidden} more cover the search, the directions,\n"
            f"the prompts and the scoring, the capability probe,\n"
            f"the machine and the output:\n"
            f"  senbonzakura --help-all\n")
    return ap


#: What `--track` means when nobody passed it. A sentinel rather than a path or the bundled alias,
#: because the right answer depends on what is on the machine and the run has to be able to SAY
#: which it chose. `resolve_defaults` turns it into one of the two.
TRACK_AUTO = "auto"
