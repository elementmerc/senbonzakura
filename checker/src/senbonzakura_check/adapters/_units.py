# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""What a foreign harness's metric is measured in, when the file does not say.

WHY THIS EXISTS, 2026-09-27

Two checks in this package are gated on `units == "proportion"`:
`impossible-proportion-reported`, which fires on a rate outside 0 to 1, and
`rate-reported-on-a-sample-too-small-to-carry-it`, which fires on a rate resting on fewer than
30 rows. Both are the checks most likely to catch a number that is simply wrong.

The lm-evaluation-harness and Inspect adapters both wrote `"units": None` for every metric they
read. So both checks were **structurally unreachable for every foreign artefact**, and the two
formats this package exists to read were the two it could say least about. A reader fed it an
lm-eval file reporting an accuracy of 1.4 and an `acc_norm` of -0.2 on a sample of 3 and got
`0 finding(s)`, exit 0.

That is worse than a missed finding. `REPRODUCING.md` invites people to point this tool at
somebody else's numbers, and the answer it gave was a clean bill of health produced by two checks
that never ran. Our own adapter set `units` correctly throughout, so the gap was invisible from
inside the project: every artefact we tested it on had the field.

WHY A NAME LIST, AND WHY A SHORT ONE

Neither format records units, so the only thing to go on is the metric's name. That makes this an
inference, and an inference in a checker has to fail towards silence: a metric wrongly called a
proportion would fire `impossible-proportion-reported` on a number that is not a proportion, which
is a false accusation about somebody else's work.

So the list holds only names whose range is 0 to 1 by definition in these harnesses, and anything
unrecognised stays `None` and goes on being skipped. Adding a name is a claim that the metric
cannot legitimately fall outside 0 to 1; if that is arguable, leave it out.
"""

#: Metric base names that are proportions by definition in lm-evaluation-harness and Inspect.
#:
#: `acc` and `acc_norm` are accuracies. `exact_match` and `em` are match rates. `f1`, `precision`
#: and `recall` are the standard set-overlap rates. `mc1` and `mc2` are TruthfulQA's multiple-choice
#: accuracies. `pass_at_1` and `pass@1` are pass rates. `accuracy` is Inspect's spelling of `acc`.
PROPORTION_METRICS = frozenset({
    "acc", "acc_norm", "acc_mutual_info", "accuracy",
    "exact_match", "em",
    "f1", "precision", "recall",
    "mc1", "mc2",
    "pass_at_1", "pass@1",
})

#: Names deliberately EXCLUDED, recorded so the next person does not have to re-derive the
#: judgement and so a well-meaning addition has to argue with this list first.
#:
#: The perplexities are unbounded above. `bits_per_byte` is unbounded above. `mcc` and
#: `matthews_corrcoef` run from -1 to 1, so a negative value is legitimate and flagging it would be
#: wrong. `bleu`, `chrf`, `ter` and the `rouge` family are reported on 0 to 100 by some harnesses and
#: 0 to 1 by others, so the same number means different things in different files and no single
#: range is safe to assert.
NOT_PROPORTIONS = frozenset({
    "perplexity", "word_perplexity", "byte_perplexity", "bits_per_byte",
    "mcc", "matthews_corrcoef",
    "bleu", "chrf", "ter",
    "rouge1", "rouge2", "rougeL", "rougeLsum",
    "brier_score",
})


def units_for(metric_name):
    """`"proportion"` when the name is one whose range is 0 to 1 by definition, else None.

    None means "this file does not say and the name does not tell us", which leaves the two
    value checks skipping rather than guessing. A skip is reported as a skip, so a reader can see
    the checker did not answer, which is the outcome this package prefers over a confident wrong
    answer.
    """
    if not isinstance(metric_name, str):
        return None
    name = metric_name.strip()
    if name in NOT_PROPORTIONS:
        return None
    return "proportion" if name in PROPORTION_METRICS else None
