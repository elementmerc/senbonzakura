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


def split_mode(argv):
    """Peel off the mode word, returning (bankai, remaining argv).

    `kageyoshi` is a real subcommand rather than an argv[0] trick: it runs the abliterator with
    the auto-scaled best-effort preset, resolved after the model loads once the architecture and
    parameter count are known. `auto` is a plain-English alias for it, not a second mode; someone
    meeting this tool for the first time should not have to know a Japanese sword release to get
    the setting that thinks for them. `abliterate` names the default explicitly.

    Here rather than in `cli` because the entry point has to know which words are modes before it
    can parse, and it must reach that answer without importing torch.
    """
    if argv and argv[0] in ("kageyoshi", "auto"):
        return True, argv[1:]
    if argv and argv[0] == "abliterate":
        return False, argv[1:]
    return False, list(argv)


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
    ap = argparse.ArgumentParser(add_help=False)
    # `model_required=False` ONLY for the abliterate parser, which also takes the model as a
    # positional so that `senbonzakura Qwen/Qwen3-1.7B` works. It still refuses a run with no
    # model at all; the check just moves from argparse to `resolve_model`, where it can say which
    # of the two ways to give it you meant to use. Every other command keeps the flag required.
    ap.add_argument("--model", required=model_required, default=None, help=model_help)
    ap.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu")
    ap.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true",
                    help="allow models that ship custom modelling code (some Hub models need "
                         "it); off by default.")
    # Omitted for a command whose measurement does not depend on prompt format, so it does not
    # offer a knob that would change nothing.
    if chat_template:
        ap.add_argument("--chat-template", dest="chat_template", default="",
                        help="a Jinja chat template for a model that ships none: either a path "
                             "to one, or the name of one this tool bundles ('plain'). Prompt "
                             "format drives every measurement here, so a missing template is an "
                             "input you supply and the run records, not something the tool "
                             "invents. A bundled one is recorded by NAME, so two runs under it "
                             "are comparable and a reader can see which format produced the "
                             "numbers.")
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


class _HelpAll(argparse.Action):
    """Print the full help, which is what `--help` printed before the split.

    A separate parser rather than a stored one: this parser has already had its help text
    suppressed by the time anybody can type the flag, and un-suppressing in place would mean
    holding two descriptions of every flag.
    """

    def __init__(self, option_strings, dest, **kw):
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kw)

    def __call__(self, parser, namespace, values, option_string=None):
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
        prog="senbonzakura",
        description="Refusal abliteration for transformer language models, with the instruments "
                    "to measure what the edit cost. A quality-guarded Optuna (NSGA-II) search "
                    "over windowed, per-component, multi-directional weight ablations.",
        epilog=(
            "commands:\n"
            "  abliterate   remove refusal directions and save the model (the default: the flags "
            "below work with or without the word)\n"
            "  kageyoshi    abliterate with the auto-scaled best-effort preset. It detects the "
            "architecture (dense / fused MoE / expert-list) and parameter count, scales the search "
            "budget and turns on every quality lever, so you set only the paths. It owns the search "
            "knobs; manual --trials / --max-directions and the rest are ignored in this mode\n"
            "  compass      measure harm discrimination as the HARMFUL/BENIGN logit margin (AUC), "
            "with the construct-validity controls beside it\n"
            "  score        refusal, hedging, the Heretic keyword rate and broken output on a "
            "fixed eval set\n"
            "  coherence    perplexity of a fixed neutral passage, the coherence cost\n"
            "  drift        one coherence ruler applied to any model after the fact, so a model "
            "edited by any tool can be measured on the same scale\n"
            "  track        build an evaluation track with a checked fit / search / measure "
            "split. `track build` fetches the prompt pools to split, from public sources at "
            "pinned revisions\n"
            "  corpora      fetch and pack the refusal corpora this tool measures with. A "
            "released wheel already carries them; an install made straight from the repository "
            "does not, because they are generated rather than committed\n"
            "  auto         alias for kageyoshi, for anyone who has not met the name\n"
            "  interactive  a guided walk through the handful of choices that decide whether a "
            "run means anything. It prints the exact command it is about to run before running "
            "it, so the second time you can type that instead\n"
            "  validate     ask whether a direction set carries refusal or carries topic: "
            "leave-one-cluster-out generalisation, a random-direction floor, and a sweep of "
            "direction count against ablation strength compared at matched refusal removal\n"
            "  capability   what the edit COST, on a task the model either gets right or does "
            "not. Refusal rates and KL cannot see reasoning loss: a model can hold a low KL with "
            "nothing broken and still have lost multi-step arithmetic. Grades against a reference "
            "answer, counts an ungradeable answer as indeterminate rather than wrong, and reports "
            "the paired change against a stock run with an interval\n"
            "  judge        check a grading model against reference labels BEFORE letting it "
            "grade anything. Reports agreement above chance rather than raw agreement, because a "
            "judge that answers the common label every time agrees 90% of the time on a 90% set "
            "and has learned nothing, and reports how much of each class it caught, because one "
            "blind to the rare class flatters whatever it grades. Exits non-zero when not "
            "certified\n"
            "  report       assemble a run's artefacts into the card that should go beside the "
            "weights. States only what the artefacts support: a rate the sample cannot carry is "
            "printed as counts, an interval spanning zero is written as not distinguishable from "
            "no change, and a section with no evidence says NOT MEASURED rather than being left "
            "out, because an absent section reads as nothing to report\n"
            "  head-to-head run this tool and other abliteration tools over the same model, the same "
            "corpus and the same budget on one machine, then report what separates them\n"
            "  convert      turn edited weights into a GGUF with the pinned converter, and "
            "optionally quantise in the same command, so an edited model becomes something "
            "llama.cpp will serve in one step\n"
            "  quantise     shrink a GGUF with the pinned llama-quantize, then read the output "
            "back to confirm it is the quantisation that was asked for\n"
            "  fetch        download a model file and prove it is the one asked for: length, GGUF "
            "header, architecture and the quantisation its name claims\n"
            "  imatrix      compute an importance matrix so a quantisation keeps the weights "
            "that matter, and record what it was calibrated on so two quantisations can be "
            "told apart\n"
            "  check        read an evaluation result file, from this tool or another one, and "
            "report how the number could be wrong: each finding names the incident behind it, "
            "the fix, and what would make the finding itself wrong. Needs no model, no corpus "
            "and no card\n"
            "  baseline     turn one measurement into the baseline file the gate compares "
            "against, carrying the conditions it was measured under: the number, its interval, "
            "the sample it rests on, the seeds, the estimator and the precision. Refuses rather "
            "than guesses when any of those is missing, and names all of them at once. Needs no "
            "model and no card\n"
            "  gate         compare a measurement against a recorded baseline and fail a build "
            "when a property moved outside its interval. Refuses rather than compares when the "
            "two were measured under different conditions, because a green tick on two numbers "
            "that never matched is worse than no gate. Needs no model and no card\n"
            "  setup        put the right build of torch on THIS machine: pip picks by platform, "
            "not by hardware, so a Windows box with a GPU gets a CPU-only wheel\n"
            "  doctor       check this install can actually do the job: pins, the binary runs, "
            "every architecture module imports, the bundled data decodes, and with --deep a real "
            "model goes through convert and quantise\n"
            "\n"
            "each command takes --help of its own, e.g. `senbonzakura compass --help`"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[loader_parser(
            model_help="HF model id or local path to abliterate",
            four_bit_help="NOT supported by the abliterator: the weight bake needs full "
                          "precision. Use it with the scorer "
                          "(python -m senbonzakura.score --load-in-4bit) to measure a model "
                          "on low VRAM.",
            model_required=False)])
    # THE ONE-COMMAND FORM: `senbonzakura Qwen/Qwen3-1.7B`, with everything else defaulted.
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
                         "Qwen/Qwen3-1.7B` is the whole command: the track, the output directory "
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
                    help="weight on KL in the SCALAR objective (higher = protect quality more). "
                         "Two things about it that the old one-line help did not say and a user "
                         "could not find out by trying. It does nothing under the default "
                         "`--search pareto`, which carries KL as its own frontier axis and never "
                         "evaluates the weighted sum, so under pareto this changes only the "
                         "`obj=` figure printed per trial and never which configuration wins. And "
                         "even under `--search scalar` the term is gated at a KL ceiling, so it "
                         "contributes nothing at all while KL stays under that ceiling. A run "
                         "that sets it and sees no difference is not being ignored by accident")
    ap.add_argument("--layer-lo", type=float, default=0.3, help="search layers from this fraction of depth")
    ap.add_argument("--layer-hi", type=float, default=0.8,
                    help="search layers up to this fraction of depth (default: 0.8). The window is "
                         "a fraction rather than a layer number so the same setting means the same "
                         "thing on models of different depths")
    ap.add_argument("--gen-tokens", type=int, default=_DEFAULT_BUDGET,
                    help=f"how many tokens each reply gets while the search is scoring it "
                         f"(default: {_DEFAULT_BUDGET}). THIS IS THE SETTING MOST LIKELY TO MAKE A "
                         f"REFUSAL RATE READ LOW, and lowering it does more damage than "
                         f"misreporting a number: the search then SELECTS for configurations "
                         f"whose refusal simply arrives after the cutoff. A refusal is only "
                         f"counted if the model gets far enough to say it, and this project "
                         f"measured its own refusal markers at a median of character 306, roughly "
                         f"token 77. The default is the point its length sweep measured the rate "
                         f"as settling on Qwen3-1.7B; that is one model, so measure yours with "
                         f"`senbonzakura score --length-sweep` rather than assuming it transfers. "
                         f"Below {_VISIBILITY_FLOOR} this command refuses unless you also pass "
                         f"--short-budget-ok")
    ap.add_argument("--short-budget-ok", dest="short_budget_ok", action="store_true",
                    help=f"allow --gen-tokens below {_VISIBILITY_FLOOR}, which is otherwise "
                         f"refused. For a smoke test or a plumbing check, where the run is not "
                         f"going to be quoted: a search at that budget selects for models whose "
                         f"refusal simply arrives after the cutoff, so the resulting numbers are "
                         f"about the budget and not about the model")
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
                         "`senbonzakura track`. Pass the word 'default' to use the evaluation "
                         "track bundled in this install, which needs no network and no download "
                         "and is the quickest way to a first run (CC BY-NC 4.0, attribution "
                         "required, non-commercial). Left out, it takes ./track when that exists "
                         "and the bundled one otherwise, and says in the log which it chose.")
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
                    help="CEILING on refusal directions per layer, not a setting for how many to "
                         "use. Each trial draws its own count between --min-directions and this, "
                         "so the search decides whether a second direction earns its place and "
                         "reports what it chose as `num_directions` in the artefact. Left alone "
                         "it picks one more often than not. Set this and --min-directions to the "
                         "same number to pin the budget, which is what turns a ceiling into an "
                         "experiment")
    ap.add_argument("--direction-clusters", type=int, default=8,
                    help="how many refusal modes to look for per layer. The harmful prompts are "
                         "clustered and each cluster proposes one candidate direction; the ones "
                         "that separate harmful from harmless best are kept, up to "
                         "--max-directions. Deliberately independent of --max-directions so the "
                         "candidate set does not change when the budget does, which is what makes "
                         "a K=1 against K=3 comparison a comparison of K.")
    ap.add_argument("--separation-statistic", dest="separation_statistic",
                    choices=_separation.CHOICES, default=_separation.DEFAULT_STATISTIC,
                    help="which statistic decides whether a candidate axis carries refusal rather "
                         "than topic. 'cohens-d' (default) is the incumbent and its threshold "
                         "means a different thing at every cluster size; 'variance-ratio' is the "
                         "ANOVA F, whose null is 1.0 at any size. Under measurement (Q-14): the "
                         "default does not change until that measurement says it should.")
    ap.add_argument("--matched-scoring", dest="matched_scoring", action="store_true", default=True,
                    help="judge each candidate direction against the harmless prompts nearest it "
                         "in content, instead of against the harmless set at large. ON BY "
                         "DEFAULT since 2026-09-17. A cluster about explosives stands out from "
                         "harmless prompts in general whether or not the model refuses it, so the "
                         "unmatched comparison cannot tell refusal from subject matter; holding "
                         "the subject still leaves refusal as the only thing that varies. This "
                         "was off pending Q-14, which has since reported: under the unmatched "
                         "comparison no statistic could tell a world containing refusal from one "
                         "containing none, and the matched one separates them by 40 to 60 points")
    ap.add_argument("--no-matched-scoring", dest="matched_scoring", action="store_false",
                    help="score candidates against the harmless set at large, as runs before "
                         "2026-09-17 did. For reproducing an older run, and for nothing else: "
                         "the comparison it restores is the one Q-14 showed cannot distinguish "
                         "refusal from subject matter")
    ap.add_argument("--capability-eval", dest="capability_eval", default="bundled",
                    help="what the capability probe measures against. 'bundled' (the default) is "
                         "256 grade-school arithmetic questions that ship with the package, so "
                         "this works offline. Give a graded benchmark instead (a question column "
                         "and an answer column, e.g. openai/gsm8k:main::test) to use your own, or "
                         "an empty string to turn the probe off. Refusal rates, the keyword rate, "
                         "drift and brokenness cannot see reasoning loss: a model can hold a low "
                         "KL with nothing broken and have lost multi-step arithmetic, because "
                         "none of them asks it to reason.")
    ap.add_argument("--capability-n", dest="capability_n", type=int, default=200,
                    help="how many items the capability probe uses (0 = off). 200 by default, and "
                         "the number is chosen rather than round: before and after are scored on "
                         "the SAME items, so the comparison is paired, and 200 resolves the "
                         "several-point drop this class of edit is reported to cause. A much "
                         "smaller sample produces a figure whose error bar covers the effect, "
                         "which reads like a measurement and is not one.")
    ap.add_argument("--capability-task", dest="capability_task",
                    choices=_capability_tasks(), default="numeric",
                    help="how the probe grades: see `senbonzakura capability --help`.")
    ap.add_argument("--capability-max-new", dest="capability_max_new", type=int, default=512,
                    help="token budget per probe answer. A worked solution is long, and a budget "
                         "that truncates them measures the budget rather than the model; "
                         "truncated answers are counted as ungradeable, never as wrong. 512 is "
                         "measured rather than guessed: on a model that reasons before answering, "
                         "256 tokens left 11 of 24 items ungradeable and 512 left 1.")
    ap.add_argument("--slow-probe-ok", dest="slow_probe_ok", action="store_true",
                    help="run the capability probe on a CPU even when it will take hours. The "
                         "defaults above are sized for a GPU, where they are minutes; measured on "
                         "a CPU with a 1.7B model they are about four hours, so the run stops and "
                         "says so rather than looking identical to a hung one for an afternoon. "
                         "Same shape as --short-budget-ok: the honest default stays, and spending "
                         "that long has to be asked for.")
    ap.add_argument("--ablation-rounds", dest="ablation_rounds", type=int, default=0,
                    help="how many times to alternate restoring the row lengths and removing the "
                         "direction again. 0 (default) is the single pass this tool has always "
                         "done, which MEASURABLY leaves part of the direction behind: restoring "
                         "the lengths undoes some of the ablation, by 5%% to 46%% depending on how "
                         "uneven the lengths are. 4 rounds removes it fully AND keeps the lengths. "
                         "Off by default because whether a cleaner cut makes a better model is an "
                         "open question and turning it on changes every number a run produces.")
    ap.add_argument("--method", choices=_methods.CHOICES, default=_methods.DEFAULT_METHOD,
                    help="which ablation recipe to run. 'searched' (default) optimises how much "
                         "to ablate and where; 'single-pass' fixes it at full strength on one "
                         "direction with no search, which is how the tools that retain capability "
                         "best are described as working; 'single-pass-raw' additionally drops the "
                         "norm restoration and exists as a control. The recipe is recorded in "
                         "abliteration.json, so two runs are comparable arms rather than two runs "
                         "whose flags a reader has to diff.")
    ap.add_argument("--free-base-model", dest="free_base_model", action="store_true",
                    help="delete the local base model directory just before writing the output, "
                         "when there is not room for both. Off by default and it always will be: "
                         "it is irreversible, it happens at the end of a long run when nobody is "
                         "watching, and the model is re-downloadable while the run is not. "
                         "Refused outright when the source is a shared Hugging Face cache, when "
                         "--out is the same directory or sits inside it, and when there is "
                         "already room. Until now this existed only as a sentence inside a disk "
                         "error, which does not help a volume that is already full.")
    ap.add_argument("--harmless-matched", dest="harmless_matched", default="",
                    help="a second harmless set, written on the SAME subjects as the harmful one, "
                         "used as the pool that --matched-scoring draws its controls from. Without "
                         "it the controls are the nearest rows of the ordinary harmless set, which "
                         "is only as good as whatever that set happens to contain on the subject. "
                         "Requires --matched-scoring, and the run refuses rather than accepting a "
                         "corpus it would not use. Whether the matching then worked is reported as "
                         "matching_quality in abliteration.json, and a value near 1.0 means it did "
                         "not.")
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
                    help="sparse surgery: fraction of output-rows to LEAVE untouched per weight, "
                         "editing only the top-magnitude (most refusal-writing) rows. 0.0 (default) "
                         "edits every row as before; e.g. 0.3 leaves the quietest 30%% of rows pristine "
                         "for less collateral. A/B against 0.0 per model to see if coherence improves "
                         "at equal refusal removal. Composes with --ablation-rounds: the rounds "
                         "honour the same mask, so the rows left out stay pristine.")
    ap.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=True,
                    help="seed the search with one sane diff-of-means config (mid-late window, full "
                         "projection, single direction) so NSGA-II/TPE begin from a known-decent point "
                         "instead of cold random sampling. On by default; --no-warm-start to A/B the "
                         "cold search.")
    ap.add_argument("--no-good-orth", action="store_true", dest="no_good_orth",
                    help="ablation study: do NOT orthogonalise the refusal direction against the "
                         "harmless mean (Refinement 3). Uses the raw difference-of-means instead. This "
                         "toggles off the projection grimjim calls 'projected abliteration'; on by "
                         "default. For measuring whether the projection helps or hurts the search.")
    ap.add_argument("--no-norm-restore", dest="no_norm_restore", action="store_true",
                    help="CONTROL ARM ONLY, and it makes a worse model on purpose. Remove the "
                         "refusal directions WITHOUT putting each weight row's original length "
                         "back. That restore is what keeps an edited model coherent, and it also "
                         "undoes part of the ablation because scaling rows does not commute with "
                         "a projection across them. This flag is the naive formulation the "
                         "restore is supposed to beat, so the difference can be measured instead "
                         "of asserted. NOT the same thing as --no-good-orth, which changes how "
                         "the directions are found rather than how they are applied.")
    ap.add_argument("--skip-conv-ablation", dest="skip_conv_ablation", action="store_true",
                    help="CONTROL ARM ONLY. Leave the output projections of non-attention "
                         "sequence mixers untouched on a hybrid architecture: a short convolution "
                         "on LFM2, a gated delta net on Qwen3.5. Those layers carry no attention "
                         "and write the residual stream through the mixer instead, and on some "
                         "models they are most of the stack (30 of 40 on Qwen3.6-35B-A3B, 18 of "
                         "24 on LFM2.5-8B-A1B). The resulting model is a PARTIAL abliteration by "
                         "construction: it exists to answer whether refusal travels through the "
                         "convolution path at all, by comparison against a run without this flag. "
                         "Every skipped layer is warned about and the choice is recorded in the "
                         "result file, so the model cannot later be mistaken for a whole one.")
    from . import events as _events
    _events.add_argument(ap)
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for the Optuna sampler (default 42). Vary it to measure run-to-run "
                         "spread: a single run tells you nothing about whether a gap between two "
                         "configurations is real. Note GPU kernels are not bit-deterministic, so a "
                         "fixed seed reproduces the search path, not the last decimal of a score.")
    ap.add_argument("--search", choices=["pareto", "scalar"], default="pareto",
                    help="pareto: NSGA-II maps the whole refusals-vs-KL frontier, we pick the knee "
                         "(intact + most uncensored). scalar: the old single weighted objective (TPE).")
    ap.add_argument("--per-component", dest="per_component", action="store_true", default=True,
                    help="tune attn.o_proj and mlp.down_proj SEPARATELY (Heretic-style). The MLP "
                         "profile may go to zero (leave the MLP untouched), which often preserves "
                         "intelligence. This is the default.")
    ap.add_argument("--uniform", dest="per_component", action="store_false",
                    help="apply ONE strength profile to both components (the pre-decouple behaviour).")
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
                    help="the FEWEST directions a trial may use. --max-directions is a ceiling "
                         "and the search picks anywhere beneath it, so 'up to two' is not 'two': "
                         "set both to the same number to pin the budget. That is what turns a "
                         "K comparison into an experiment rather than a mixture, and a run on "
                         "2026-08-12 chose one direction on three seeds of five when left free.")
    ap.add_argument("--max-kl", dest="max_kl", type=float, default=None,
                    help="the most coherence drift you will accept, as KL. Sets both the hard "
                         "intactness filter and where the knee's coherence surcharge begins, so "
                         "the search returns the biggest refusal reduction it can manage UNDER "
                         "this figure rather than wherever the frontier's knee happens to sit "
                         f"(default: filter at {KL_CEIL}, surcharge above {KL_TARGET}). If no "
                         "configuration meets it the run refuses rather than quietly returning "
                         "one that does not. Heretic's comparable setting defaults far tighter, "
                         "so this is the flag that puts the two tools at one operating point.")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop the search early if no trial improves the best scalarised score for "
                         "this many consecutive trials (0 = run all --trials).")
    ap.add_argument("--eval-refusal-final", type=int, default=128,
                    help="re-score the top frontier candidates on this many bad-eval prompts, "
                         "held out from the ones the search scored against, before picking the "
                         "knee. ON BY DEFAULT since 2026-09-17, at 128. Without it the winner is "
                         "chosen on the same small set every trial was scored on, so it is the "
                         "best of N draws over those particular prompts rather than a "
                         "measurement, and the figure the run reports describes the rows it was "
                         "selected on. The cost is one extra scoring pass over --top-rescore "
                         "candidates, which is minutes against a search measured in hours. Pass 0 "
                         "to skip it and use the search-eval numbers, which is what runs before "
                         "this date did")
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
                    help="show every flag, with the full description of each. This page shows the "
                         "ones a run needs; there are 69 in total, and the rest are the search, "
                         "the scoring and the measurement knobs.")
    if not full:
        # AT THE END, ON WHAT WAS ACTUALLY DECLARED, so a flag added later cannot escape the
        # split by being added somewhere this function does not look.
        hidden = 0
        for action in ap._actions:
            if action.dest not in CORE_FLAGS and action.help is not argparse.SUPPRESS:
                action.help = argparse.SUPPRESS
                hidden += 1
        ap.epilog += (
            f"\n\nThis page shows the {len(ap._actions) - hidden} flags a run needs. "
            f"{hidden} more control the search, the scoring and the measurement:\n"
            f"  senbonzakura --help-all\n")
    return ap


#: What `--track` means when nobody passed it. A sentinel rather than a path or the bundled alias,
#: because the right answer depends on what is on the machine and the run has to be able to SAY
#: which it chose. `resolve_defaults` turns it into one of the two.
TRACK_AUTO = "auto"
