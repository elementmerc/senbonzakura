# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""One row from a public leaderboard, read as a set of scores with nothing behind them.

WHY A ROW IS WORTH READING AT ALL

A public ranking is, for several model publishers, the only number about their work that anybody
outside sees. It is also the number they get judged on. So the useful question is not "what is the
score" but "what can this score support", and for every public ranking of this kind the answer is
less than a reader assumes.

Measured on the UGI Leaderboard's own published data file on 2026-10-05: **1,326 rows, 71 columns,
and not one confidence interval, standard error or sample size among them.** That absence is the
finding. It is not a criticism of the leaderboard, whose confidential question set is a defensible
choice that stops publishers training against the test; it is a statement about what a reader may
conclude from a row of it.

THE DISCIPLINE THIS FILE IS BUILT AROUND, AND IT IS A REFUSAL

**It does not compute an interval it has no sample size for.** Given a score of 36.5 and no n,
there is no interval, and inventing one by assuming a plausible n would be exactly the fabrication
this whole package exists to catch in other people's work. The correct output is the sentence "this
number cannot be given an interval because the source publishes no sample size", and it is more
useful than a number, because a reader who sees an interval believes the figure has been checked.

So every metric here is normalised with `n: None` and `stderr: None`, truthfully, and the checks
that need either of those skip rather than guess.

WHY IT TAKES A ROW THE USER SUPPLIES AND FETCHES NOTHING

A checker that phones a third party breaks when they move a URL, and it drags a network dependency
into a subpackage whose entire selling point is that it needs nothing. So the input is a CSV the
user already has, or a single row they pasted. Nothing here opens a socket.

AN INTERACTION WITH AN EXISTING CHECK, FOUND BY RUNNING IT AND NOT BY DESIGN

Because every metric here truthfully carries no denominator, `null-reported-as-zero` fires on any
column whose value is zero. On a real row it fired on `Avg Thinking Chars: 0.0` and
`Repetition Interrupts: 0.0`.

That was examined rather than suppressed, and it is a true positive. A reader of the row cannot
tell whether a zero there means "measured, and it was zero" or "not applicable to this model, so
nothing was measured". For a non-thinking model the thinking-character count is the second, and the
ranking records the distinction in a separate `Is Thinking Model` column rather than in the cell, so
the cell on its own is genuinely ambiguous. Conflating not-applicable with zero is exactly the class
of thing this checker exists to point at, so the finding stays.

HOW A SCORE IS TOLD FROM A LABEL

The column set of a public leaderboard is not knowable in advance and will change without notice,
so there is no hardcoded list of score columns. Any column whose value parses as a number is a
score; everything else is an attribute. That is deliberately dumb, because the alternative is a
list that silently stops covering a leaderboard the week they add a column.
"""
from __future__ import annotations

#: Stamped by the loader, for the same reason as the GGUF adapter: a CSV row is not a JSON
#: document, and `detect` only sees dicts.
FORMAT_KEY = "artefact_format"
FORMAT_VALUE = "leaderboard-row"

#: Columns that name the thing being scored rather than scoring it. Matched case-insensitively on
#: the stripped name. Kept short: anything not listed and not numeric is carried as an attribute,
#: so a column this list has never heard of is never mistaken for a score.
_IDENTITY_COLUMNS = {
    "author/model_name", "model", "model_name", "model_display", "modelid", "model id",
    "name", "author", "organisation", "organization", "url", "model link", "link",
}

#: Attributes worth lifting by name because a check can read them. Everything else still travels
#: in `attributes`.
_KNOWN_ATTRIBUTES = {
    "release date": "release_date",
    "test date": "test_date",
    "prompt template": "prompt_template",
    "architecture": "architecture",
    "total parameters": "total_parameters",
    "active parameters": "active_parameters",
    "is finetuned": "is_finetuned",
    "is merged": "is_merged",
    "is foundation": "is_foundation",
}


def _as_number(value):
    """A float, or None. Total: never raises, whatever the cell holds."""
    if isinstance(value, bool):
        # A bool is an int in Python and a flag in a leaderboard. Scoring it would turn
        # "Is Finetuned" into a metric worth 1.0.
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().replace(",", "")
    if not text or text in {"-", "--", "n/a", "N/A", "null", "None", "?"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def quoted_precision(value) -> int | None:
    """How many decimal places the source quoted, or None when it is not a quoted decimal.

    Needed because a difference smaller than the precision a number was printed at is not a
    difference anybody can read off the page. `36.5` against `36.51` is a comparison the first
    figure cannot participate in.

    Read off the STRING rather than the float, because `36.50` and `36.5` are the same float and
    are different claims about precision.
    """
    if isinstance(value, str):
        text = value.strip()
        if "." in text and _as_number(text) is not None:
            return len(text.rsplit(".", 1)[1])
        return 0 if _as_number(text) is not None else None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return 0
    return None


class LeaderboardRowAdapter:
    name = "public leaderboard row"

    @staticmethod
    def detects(doc) -> bool:
        return doc.get(FORMAT_KEY) == FORMAT_VALUE

    @staticmethod
    def normalise(doc) -> dict:
        row = doc.get("row") or {}
        source = doc.get("source")

        model = None
        attributes: dict = {}
        metrics: dict = {}

        for raw_key, raw_value in row.items():
            key = str(raw_key).strip()
            low = key.lower()
            if low in _IDENTITY_COLUMNS:
                if model is None and isinstance(raw_value, str) and raw_value.strip():
                    model = raw_value.strip()
                attributes[key] = raw_value
                continue
            if low in _KNOWN_ATTRIBUTES:
                attributes[key] = raw_value
                continue
            number = _as_number(raw_value)
            if number is None:
                attributes[key] = raw_value
                continue
            metrics[key] = {
                "metric": key,
                "value": number,
                # The source of the figure is the leaderboard, and naming it as the estimator is
                # the honest answer: there is no published method behind the column beyond the
                # leaderboard's own, and no two leaderboards' columns are interchangeable.
                "estimator": f"{source} column" if source else "leaderboard column",
                # NO UNITS CLAIMED. A column called `UGI` on a 0 to 100 scale and one called
                # `W/10` on a 0 to 10 scale sit side by side, and nothing in the file says which
                # is which, so asserting `proportion` would make the impossible-proportion check
                # fire on every row of a 0 to 100 column. Left absent so that check skips.
                "units": None,
                # THE TWO REFUSALS THIS ADAPTER EXISTS FOR. Both truthfully None: the source
                # publishes neither, and filling either one from an assumption is the fabrication
                # the module docstring forbids.
                "n": None,
                "stderr": None,
                "quoted_decimals": quoted_precision(raw_value),
            }

        return {
            "artefact_kind": "leaderboard-row",
            "leaderboard": source,
            "model": model,
            "metrics": metrics,
            "attributes": attributes,
            "metric_count": len(metrics),
            # Flat true for this artefact kind by construction, and stated as a field so a check
            # can read it rather than having to know the kind. Every score here carries no sample
            # size, which is what makes the row uncomparable with another row.
            "any_metric_carries_n": False,
            "any_metric_carries_uncertainty": False,
            # Deliberately absent rather than guessed: a leaderboard row says nothing about which
            # rows the score came from, which is precisely the gap.
            "eval_split": None,
            "limit": None,
        }


__all__ = ["FORMAT_KEY", "FORMAT_VALUE", "LeaderboardRowAdapter", "quoted_precision"]
