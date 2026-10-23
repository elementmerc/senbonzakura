# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Arguments a command can work out for itself, resolved in one place.

WHY A MODULE RATHER THAN A FUNCTION IN EACH COMMAND

Two spellings of the model (`senbonzakura capability MODEL` and `--model MODEL`) have exactly two
failure modes, and both of them have to be refused rather than resolved by precedence: neither
given, and both given and disagreeing. A per-command copy of that logic is a per-command chance to
silently prefer one, and an artefact recording the winner with nothing saying the other was ever
mentioned is the specific failure this project has already had to withdraw results over.

WHY NOT IN `parser.py`

That module DECLARES flags; these READ them. `tools/research/audit_flags.py` is built on the
split, and a declaring module that also reads its own flags breaks the pairing for every flag in
the file. That is not a style preference: putting two of these in `parser.py` reported all 66
flags as unread.

WHAT MAY BE IMPORTED HERE

Nothing heavy. `cli.py` imports torch at module scope, so a measurement command reaching these
through it would need the whole abliteration stack in order to parse a command line, which is the
defect `entry.py` exists to prevent. The standard library is the whole budget.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

#: The flags where a negative value would genuinely slice a sequence from its end, which is the
#: hazard this validator was written for: `score --skip -5` scored the LAST five prompts and
#: stamped the partition `rows-from--5`, a boundary that does not exist.
#:
#: WHY A SET RATHER THAN THE DEFAULT, 2026-10-13. That explanation was printed for every flag,
#: and it is false for most of the ninety places this validator is used. A seed does not slice
#: anything. Neither does a batch size, an optimiser step count, a token budget, a thread count or
#: a direction cap. `senbonzakura tamper --seed -1` was told that a negative seed "slices from the
#: END of the set, so the run would measure rows nobody asked for", which is a confident
#: explanation of a mechanism that is not there. Found by journey 2, which met it on three flags;
#: it was wrong on far more than three.
#:
#: Named here rather than passed at each call site because there are ninety of them, and a reason
#: that has to be repeated ninety times is a reason that will be wrong somewhere within a month.
SLICES_ROWS = frozenset({
    "--n", "--skip", "--skip-harmful", "--skip-harmless", "--skip-matched", "--train-n",
    "--dir-prompts", "--eval-refusal", "--eval-refusal-final", "--eval-kl", "--trials",
    "--capability-n", "--probe-n", "--top-rescore",
})


def calendar_date(what):
    """An argparse `type=` for a day written YYYY-MM-DD, kept as that text.

    The check is `firsttoken.check_template_date`, imported when a date is actually given so that a
    help page, which never gives one, still doesn't load the machine-learning libraries.
    """
    def parse(value):
        from .firsttoken import check_template_date
        try:
            check_template_date(value)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"{what}: {e}.") from None
        return value
    return parse


def whole_number(what, *, minimum=0):
    """An argparse `type=` for a count or an offset: a whole number, never below `minimum`.

    `minimum` is 0 for the counts where zero means something (`--n 0` is every prompt, `--skip 0`
    is the head) and 1 for the ones where it does not: a batch of zero is `range(0, n, 0)`, which
    is a ValueError several screens from the flag that caused it, and a token budget of zero
    measures the budget rather than the model.

    WHY AT PARSE TIME, AND WHY SHARED. `score --skip -5` passed every check it met: it is truthy,
    it is not `>= len(prompts)`, and `prompts[-5:]` is a perfectly good slice. So a refusal rate
    came back measured over the LAST five prompts, and the partition it was stamped with read
    `rows-from--5`, a boundary that does not exist. `--n -5` dropped the last five instead of
    taking the first five. Neither said anything. `capability` refused the same input correctly,
    which is the worse half of the finding: two sibling commands disagreed about whether a
    negative sample size is a thing, and the one that disagreed silently is the one people script.

    A number is checked where it is read, so a command cannot forget to ask, and the sentence
    names the flag rather than the type: argparse's own message for a bad `int` says the value is
    invalid without saying what would have been valid.
    """
    def parse(value):
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(
                f"{what} wants a whole number and got {value!r}.") from None
        if number < minimum:
            # The slicing explanation only where it is true. See `SLICES_ROWS`.
            why = (" A negative one slices from the END of the set, so the run would measure rows "
                   "nobody asked for and report them under the rows they did."
                   if what in SLICES_ROWS else "")
            raise argparse.ArgumentTypeError(
                f"{what} is {number}, and it cannot be below {minimum}.{why} Pass {minimum} or "
                f"more.")
        return number
    return parse


def real_number(what, *, minimum=None, maximum=None):
    """An argparse `type=` for a measurement: a real number inside a stated range.

    The companion to `whole_number` for the handful of flags that are genuinely continuous: a
    tolerance, a fraction, a weight. Same reasoning and the same refusal shape, so a reader meets
    one voice whichever flag they mistyped.

    `None` on either side means unbounded there, and a bound is declared only where a value past
    it is MEANINGLESS rather than merely large. A negative tolerance is not a strict comparison,
    it is a bar nothing can clear, and a run that quietly reports nothing matched is worse than
    one that refuses the flag.
    """
    def parse(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(
                f"{what} wants a number and got {value!r}.") from None
        if not math.isfinite(number):
            raise argparse.ArgumentTypeError(
                f"{what} is {value!r}, which is not a finite number. Every comparison against it "
                f"would be false, so the run would report nothing and look like it had measured.")
        if minimum is not None and number < minimum:
            raise argparse.ArgumentTypeError(
                f"{what} is {number:g} and cannot be below {minimum:g}.")
        if maximum is not None and number > maximum:
            raise argparse.ArgumentTypeError(
                f"{what} is {number:g} and cannot be above {maximum:g}.")
        return number
    return parse


def pick_model(positional, flag, *, command="senbonzakura", long_form=None):
    """One model out of two spellings, or a refusal naming both ways to give one.

    TAKES THE TWO VALUES RATHER THAN THE NAMESPACE, deliberately. Reading `args.model_positional`
    here would make this module the reader of a flag it does not declare, and
    `tools/research/audit_flags.py` pairs every declaration with the module that reads it: a
    helper reaching into a namespace it does not own breaks that pairing for the declaring module
    and reports its flags as unkept. It reported exactly that, the first time this was written as
    a namespace reader. Each command keeps the one line that reads its own arguments.

    `command` appears in the refusal, so the sentence a person reads names what they typed rather
    than the tool in general, and `long_form` carries that command's own worked example.
    """
    if positional and flag and positional != flag:
        raise SystemExit(
            f"{command}: two different models were given: {positional!r} as a positional and "
            f"{flag!r} with --model. Pass one.")
    model = flag or positional
    if not model:
        raise SystemExit(
            f"{command}: no model given.\n"
            f"  The short form is the model on its own:\n"
            f"    {command} Qwen/Qwen2.5-0.5B-Instruct\n"
            f"  The long form still works and is what a script should use:\n"
            f"    {long_form or f'{command} --model Qwen/Qwen2.5-0.5B-Instruct'}")
    return model


def refuse_to_overwrite(path, *, what="result", flag="--out"):
    """Stop before replacing a file the user did not name.

    THE REASON THIS EXISTS ALONGSIDE THE DEFAULTS. Giving `--out` a default removes a flag from
    the command line and introduces the chance of a second run quietly replacing the first run's
    numbers, which is a worse trade than the one it was meant to fix. So a default output that is
    already occupied is refused, and an output the user named explicitly is theirs to overwrite:
    somebody who typed the path has said which file they mean.
    """
    if os.path.exists(path):
        raise SystemExit(
            f"{path} is already there, and it was not named on the command line: it is the "
            f"default {what} path.\n"
            f"  Replacing it would leave two runs' numbers indistinguishable.\n"
            f"  Pass {flag} <path> to say where this run's {what} goes, or move the old one.")
    return path


def explain_that_the_output_is_positional(ap, *, takes):
    """Answer `--out` on the two commands where the output path is a positional instead.

    Nine commands in this tool spell the output `--out`; `convert` and `quantise` take it as their
    second positional, because both of them also carry flags that begin with those letters. Those
    two parsers therefore run with `allow_abbrev=False`, without which argparse prefix-matches
    `--out` to `--outtype` or `--output-tensor-type` and reports a flag the user never typed.

    Turning abbreviation off fixes the misattribution but leaves the reader with "unrecognized
    arguments: --out", which is true and tells them nothing about what to type instead. Somebody
    who learned `--out` on `senbonzakura run` is the likeliest person to meet this message, so it
    names the form that works.
    """
    refuse = ap.error

    def error(message):
        typed_it = any(word == "--out" or word.startswith("--out=") for word in message.split())
        if typed_it:
            message += (f"\n  On this command the output path is the second positional argument, "
                        f"not a flag:\n    {ap.prog} {takes} <output>")
        refuse(message)

    ap.error = error
    return ap


def shortest_invocation(parser):
    """The smallest command line this parser accepts, as a string, or None if it needs nothing.

    WHY A SUBCOMMAND OWES THIS AND argparse DOES NOT GIVE IT. argparse's refusal for the commonest
    mistake there is reads `the following arguments are required: --baseline, --current`, which
    names what is missing and nothing about what to type. Readers 3 and 5 both came to set
    something up and stopped there, because the next move after that sentence is to go and read
    `--help` and work out the shape for yourself.

    Built from the parser's own required actions rather than written per command, so the twenty
    delegated commands get it at once and a flag that becomes required later is included without
    anybody remembering. The value shown is the `metavar` where a command declared one, which is
    the word that command's own help uses, and the uppercased `dest` otherwise.

    POSITIONALS COME FIRST, in declaration order, because that is the order they have to be typed
    in. Flags are sorted, because their order does not matter and two runs should print one line.
    """
    positionals, flags = [], []
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            continue
        if not action.option_strings:
            # A positional with `nargs="?"` is optional however `required` reads, and argparse
            # does not set `required` on positionals at all.
            if action.nargs not in ("?", "*"):
                positionals.append(action.metavar or action.dest.upper())
            continue
        if action.required:
            value = "" if action.nargs == 0 else f" {action.metavar or action.dest.upper()}"
            flags.append(f"{action.option_strings[0]}{value}")
    parts = positionals + sorted(flags)
    if not parts:
        return None
    return f"{parser.prog} {' '.join(parts)}"


def unknown_long_flags(parser, argv):
    """Every `--flag` in `argv` that this parser does not declare.

    Only `--` prefixed tokens, which is what makes the check safe. A single dash could be a negative
    number (`--threads -1` is fine and `-1` is its value), and a value that merely CONTAINS a double
    dash (`--tensor-type attn_v=--x`) does not start with one. A token that starts with `--` and is
    not a declared option is what argparse itself would reject, so this agrees with argparse rather
    than second-guessing it.
    """
    known = {opt for action in parser._actions for opt in action.option_strings}  # noqa: SLF001
    return [tok for tok in argv
            if tok.startswith("--") and tok.split("=", 1)[0] not in known]


class ParserThatNamesUnknownFlags(argparse.ArgumentParser):
    """An `ArgumentParser` that mentions a mistyped flag even when a required one is also missing.

    WHAT PROMPTED IT, 2026-09-27

    `senbonzakura baseline /tmp --sedes 42` reported only `the following arguments are required:
    --seeds`. The typo was never mentioned, so the obvious next move is to add `--seeds` and run
    again, which now carries a correct flag AND a silently ignored one. A surface audit found this on
    every subcommand that has a required flag.

    The ordering is inside argparse: `parse_known_args` raises the required-argument error before it
    returns anything about unrecognised tokens, so there is nothing to reorder at the call site.

    STRICTLY ADDITIVE, which is the whole design. It never rejects anything that parses today; it
    only adds a sentence to a refusal that was already happening. The operator chose this over a
    pre-parse pass that refuses the unknown flag first (2026-09-27), because a scan that refuses can
    be wrong about a legitimate value and this cannot. The stronger version is noted for v0.5.

    IT DECLINES ON A PARSER WITH SUBCOMMANDS, deliberately. There, `argv` holds the subcommand's own
    flags, which the top-level parser does not declare and must not report as typos. argparse hands
    the subparser its own slice of argv, so the subparser still gets this behaviour where it counts.
    """

    def parse_known_args(self, args=None, namespace=None):
        # The only way to know what was parsed: argparse keeps no record, and `error` needs one.
        self._argv_as_given = list(sys.argv[1:] if args is None else args)
        return super().parse_known_args(args, namespace)

    def error(self, message):
        if "required" in message and not self._has_subcommands():
            from . import say

            unknown = unknown_long_flags(self, getattr(self, "_argv_as_given", []))
            if unknown:
                named = ", ".join(unknown)
                verb = "is not a flag" if len(unknown) == 1 else "are not flags"
                message += "\n" + "\n".join(say.lines(
                    f"Also, {named} {verb} this command has, so adding the missing one above would "
                    f"leave it silently ignored. Check the spelling against --help.",
                    indent="  ", first="  "))
            # AND WHAT TO TYPE, added 2026-10-10 for the ledger item behind readers 3 and 5. Naming
            # what is missing is not the same as showing the shape, and the next move after
            # argparse's own sentence is to go away and read `--help`. The shortest accepted
            # invocation is four spaces in so it stays pasteable: `say.reflow` passes a four-space
            # line through verbatim, which is the same rule `baseline`'s hand-written refusal uses.
            #
            # STRICTLY ADDITIVE, like the paragraph above it and for the same reason: this never
            # refuses anything that parses today, it adds to a refusal that was already happening.
            #
            # It does NOT close with its own `--help` pointer. It used to, and `_refuse` below now
            # ends every refusal with one, so keeping both printed the same sentence twice in a row.
            shortest = shortest_invocation(self)
            if shortest:
                message += "\n" + "\n".join([
                    *say.lines("The shortest command line this accepts:",
                               indent="  ", first="  "),
                    f"    {shortest}",
                ])
        self._refuse(message)

    def _refuse(self, message):
        """Say what was wrong, first, and point at the help instead of reprinting it.

        WHY THIS OVERRIDES ARGPARSE. `ArgumentParser.error` prints the usage block and then the
        error. On this tool's main parser that is thirteen lines of bracketed flag soup above the
        one line saying what the reader got wrong, and a journey persona reported exactly that:
        they read the last line, which is the error, and had to scroll back up past the wall to be
        sure nothing else was in it.

        Printing the usage AFTER the error is worse, not better: a terminal scrolls, so the thing
        a reader sees is the last thing written, and that would bury the error from the other
        side. So the block goes and a pointer replaces it. Nothing is lost: `--help` prints the
        usage, the flags and the command list, and this names it.

        Exit 2 and stderr are unchanged, because a script reading either is reading a contract.

        It also spells the refusal in British English and offers the nearest real flag. argparse
        writes "unrecognized arguments", which is the one American spelling this tool emits in a
        user-facing string, and "unrecognized arguments: --modle" is true and unhelpful next to
        `--model` sitting one letter away. The subcommand path has suggested a near match since
        `entry._not_a_command`; the flag path had not.
        """
        from . import say

        message = message.replace("unrecognized", "unrecognised")
        tail = [say.lines(f"`{self.prog} --help` lists what this command takes.",
                          indent="  ", first="  ")]
        near = self._nearest_flags(message)
        if near:
            tail.insert(0, say.lines(f"Did you mean {', '.join(near)}?",
                                     indent="  ", first="  "))
            tail.insert(1, [""])
        out = [f"{self.prog}: error: {message}", ""]
        for block in tail:
            out += block
        self.exit(2, "".join(f"{line}\n" for line in out))

    def _nearest_flags(self, message):
        """The declared flags closest to the unknown ones this refusal names, at most three.

        Reads the flags back out of the message rather than out of `argv`, so it only ever
        suggests a replacement for a token argparse itself has just rejected. A cutoff of 0.8 is
        the same one `entry._not_a_command` uses, and it is deliberately tight: a wrong suggestion
        costs more than no suggestion, because the reader will try it.
        """
        import difflib

        # `:.` as well as the comma, because argparse writes the flag with a colon after it in
        # "argument --steps: ...". Left as a bare comma, `--steps:` matched nothing in the known
        # set, survived the filter below and was then suggested as the near match for itself.
        typed = [tok.rstrip(",:.") for tok in message.split()
                 if tok.startswith("--") and len(tok) > 2]
        if not typed:
            return []
        known = sorted({opt for action in self._actions
                        for opt in action.option_strings if opt.startswith("--")})
        # A FLAG THAT EXISTS NEEDS NO SUGGESTION, 2026-10-13. The reasoning above holds only for a
        # message about an UNKNOWN flag, and argparse writes the flag's own name into a type error
        # too: "argument --steps: --steps is 0, and ...". So `senbonzakura tamper --steps 0` was
        # answered with "Did you mean --steps?", which suggests the thing the reader has just
        # typed and reads as the tool not having understood them. Found by journey 2.
        typed = [tok for tok in typed if tok.split("=", 1)[0] not in known]
        if not typed:
            return []
        out = []
        for tok in typed:
            for hit in difflib.get_close_matches(tok.split("=", 1)[0], known, n=1, cutoff=0.8):
                if hit not in out:
                    out.append(hit)
        return out[:3]

    def _has_subcommands(self):
        return any(isinstance(a, argparse._SubParsersAction)  # noqa: SLF001
                   for a in self._actions)


#: Device kinds this tool accepts. `cpu` and `cuda` are what the documentation offers; the rest are
#: accelerators torch names, accepted because somebody with one should not be told their hardware is
#: a typo. Anything else is a typo, and saying so before a download is the whole point.
KNOWN_DEVICES = frozenset({"cpu", "cuda", "mps", "xpu", "hpu", "npu", "mtia", "meta"})


def refuse_unusable_device(device, *, offload_dir=None):
    """Refuse a device this machine cannot use, for any command, before anything is downloaded.

    `offload_dir` IS TAKEN ONLY TO GET THE ADVICE RIGHT, 2026-10-13. The CUDA refusal below told
    every reader to "run on the processor instead: --device cpu", which is correct advice unless
    they also passed `--offload-dir`: a CPU placement is a plain copy that accelerate never sees,
    so it cannot spill, and the loader raises on the pair. A reader on a machine with no card who
    wanted disk offload was therefore sent from a clear refusal into a traceback by this tool's own
    suggestion. Found by the streaming journey on 2026-10-13, on a machine with no usable card.

    WHY THIS LIVES HERE RATHER THAN IN `cli`, 2026-10-11.

    `cli._preflight_device` has done exactly this since 2026-09-16, and the comment it carries
    says the lesson plainly: the failure it prevents is the one that "`entry.py` goes to some
    trouble to prevent for a missing import, arriving through the one input nobody checked". It
    was wired into the abliterate path and the guided mode, and into nothing else.

    So `senbonzakura compass --device cuda` on a machine with no card still fetched the weights,
    loaded them, began scoring, and died inside `margin.py` on `.to(device)` under fourteen frames
    of torch. **The same defect, in the same shape, one command away from its own fix.** Found by
    the constrained-user journey of 2026-10-11, whose whole point is that paying for a 3.8 GB
    download and then being told the obvious is the difference between using this tool and not.

    It is deliberately NOT inside the model loader, which is where the dependency pre-flight went.
    That one belongs there because it cannot be predicted from the command line; this one is
    decidable from a flag, dozens of tests drive the loader directly with a device string, and a
    refusal there would be a refusal in the middle of somebody's unit test rather than at the
    surface where a person typed the flag.

    `torch` is imported inside the function on purpose: this module is on the `--help` path, which
    is measured and must not pay for torch.
    """
    device = str(device or "").strip().lower()
    if not device or device.startswith("cpu"):
        return device

    import torch

    from . import say
    kind = device.split(":", 1)[0]
    if kind == "cuda" and not torch.cuda.is_available():
        # THROUGH `say.refusal_text`, which wraps to the terminal, rather than hand wrapped.
        # This was built with explicit newlines at about a hundred and twenty columns, so the
        # reason line ran off the edge of any ordinary window, and it runs off it twice now that
        # the guided mode quotes a failing check's refusal on its pre-flight board. Found by a
        # journey driving the walk with `--device cuda` on a machine with no card.
        #
        # The two command lines keep their own indentation: `say` leaves an indented line alone
        # deliberately, because an indented line is something somebody has to paste.
        raise SystemExit(say.refusal_text(
            "senbonzakura: --device is 'cuda' and this machine has no usable CUDA device.",
            "Checked before the model is downloaded, because finding out afterwards costs the "
            "download and tells you the same thing.",
            ("What to do:\n"
             "    run on the processor instead:  --device cpu   (correct, and much slower)\n"
             "    or fix the install:            senbonzakura setup --apply"
             if not offload_dir else
             "What to do:\n"
             "    drop BOTH flags:               --device cpu, and no --offload-dir\n"
             "    or fix the install:            senbonzakura setup --apply\n"
             "  --offload-dir cannot be kept here. Spilling to disk is something accelerate does "
             "while it places a model across a card, host memory and disk, and a processor run is "
             "placed by a plain copy it never sees. So on the processor the whole checkpoint has "
             "to fit in memory, and a model too large for that needs a card rather than a "
             "different flag."),
            "`senbonzakura doctor` reports the same thing as an advisory and runs without a "
            "card."))
    if kind == "mps" and not getattr(getattr(torch.backends, "mps", None), "is_available",
                                     lambda: False)():
        raise SystemExit(
            "senbonzakura: --device is 'mps' and this build of torch reports no MPS device.\n"
            "  Use --device cpu, or install a torch built with MPS support.\n"
            "  `senbonzakura doctor` reports what this install can and cannot do.")
    # AND FINALLY, WHETHER IT IS A DEVICE AT ALL. Everything above asks whether a KNOWN device is
    # usable; nothing asked whether the string names a device, so `--device banana` fell straight
    # through to `model.to("banana")` and reached the user as fourteen frames of traceback out of
    # torch, AFTER the model had downloaded and loaded. Found by a surface audit on 2026-09-27, and
    # it is the one flag in this pre-flight block that was not checked here: `--track`, `--out`,
    # `--trials`, the direction budget and CUDA availability all are.
    #
    # The accepted set is what `--help` documents, plus the accelerators torch names that somebody
    # may legitimately have. An unknown one is refused with the list rather than passed on, because
    # torch's own message is a comma-separated dump of twenty backend names with no advice in it.
    if kind not in KNOWN_DEVICES:
        raise SystemExit(say.refusal_text(
            f"senbonzakura: --device {device!r} is not a device this tool knows.",
            "Checked here rather than at load time, because torch raises on it only after the model "
            "has been downloaded and placed, and then says so in twenty frames.",
            "What to pass:\n"
            "    --device cpu       every machine, correct and slow\n"
            "    --device cuda      an NVIDIA card\n"
            "    --device cuda:1    a specific card when there are several",
            "`senbonzakura doctor` says what this machine actually has."))
    return device
