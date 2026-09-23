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

import os


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
            f"    {command} Qwen/Qwen3-1.7B\n"
            f"  The long form still works and is what a script should use:\n"
            f"    {long_form or f'{command} --model Qwen/Qwen3-1.7B'}")
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
