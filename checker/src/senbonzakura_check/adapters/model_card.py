# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A published model card, read as a set of claims about a model.

WHAT THIS REACHES THAT NOTHING ELSE DOES

Of the organisations this project would like to be useful to, ten publish a card carrying claims
and **none publishes an evaluation-harness log**. So the three original adapters, which read
harness logs, reach almost nobody, and this one reaches almost everybody. That asymmetry is the
whole reason it exists and it is worth stating plainly: the widest-reach capability in the package
was the artefact nothing could read.

WHAT IT PUTS IN THE CANONICAL VOCABULARY, AND WHAT IT REFUSES TO

A card's numbers become `metrics` entries so the existing checks can see them, with one hard rule:
**a percentage with no denominator gets `n: None`**, truthfully, and no interval. A card that says
"100%" and nowhere says of how many has published a number that cannot be given an interval, and
saying so is the correct output. Inventing a plausible denominator to produce one would be the
fabrication this package exists to catch in other people's work.

Units are claimed as `proportion` only for a fraction, where the card itself supplies both halves
and the quantity is unambiguous. A bare percentage is left without units, because a card's `%` is
as often a relative change as a rate, and asserting `proportion` would make the
impossible-proportion check fire on a line reading "16% faster".
"""
from __future__ import annotations

# Absolute rather than `from .. import`: the lint rule bans reaching into the parent
# package relatively, and the sibling convention here (`from ._units import ...`)
# does not apply to a module that lives one level up.
from senbonzakura_check import cardread

#: Stamped by the loader. A card is Markdown, not a JSON document, so it arrives the same way the
#: GGUF and leaderboard artefacts do.
FORMAT_KEY = "artefact_format"
FORMAT_VALUE = "model-card"


class ModelCardAdapter:
    name = "model card"

    @staticmethod
    def detects(doc) -> bool:
        return doc.get(FORMAT_KEY) == FORMAT_VALUE

    @staticmethod
    def normalise(doc) -> dict:
        card = doc.get("card") or {}
        claims = card.get("claims") or []

        metrics = {}
        for i, claim in enumerate(claims):
            if claim["kind"] == "fraction":
                key = f"claim{i}.{claim['count']}_of_{claim['total']}"
                units, n = "proportion", claim["total"]
            else:
                key = f"claim{i}.percent"
                # NO UNITS AND NO DENOMINATOR. See the module docstring: a card's bare percentage
                # is as often "16% faster" as it is a rate, and it carries no sample size at all.
                units, n = None, None
            metrics[key] = {
                "metric": "claimed_score",
                "value": claim["value"],
                # The card is the estimator, which is the honest answer: there is no stated
                # procedure behind the figure beyond "the publisher reported it".
                "estimator": "stated in a model card",
                "units": units,
                "n": n,
                "stderr": None,
                # Carried so a reader can see what the publisher was actually talking about,
                # rather than the parser guessing a benchmark name for somebody else's figure.
                "context": claim["context"],
                "saturated": claim["saturated"],
            }

        return {
            "artefact_kind": "model-card",
            "model": card.get("card_name"),
            "metrics": metrics,
            "declared_licence": (card.get("frontmatter") or {}).get("license"),
            "declared_base_model": (card.get("frontmatter") or {}).get("base_model"),
            "claim_count": card.get("claim_count"),
            "absolute_claims": card.get("absolute_claims") or [],
            "absolute_claim_count": card.get("absolute_claim_count"),
            "absolute_claim_without_any_number": card.get("absolute_claim_without_any_number"),
            "saturated_total": card.get("saturated_total"),
            "saturated_context": card.get("saturated_context"),
            "saturated_interval_low": card.get("saturated_interval_low"),
            "saturated_interval_high": card.get("saturated_interval_high"),
            "lowest_indistinguishable_count": card.get("lowest_indistinguishable_count"),
            "lowest_indistinguishable_share": card.get("lowest_indistinguishable_share"),
            "indistinguishable_gap_pp": card.get("indistinguishable_gap_pp"),
            # A ONE-ENTRY MAPPING, so the check can use `any_outside` and keep its threshold in
            # the check file where a reader can audit it. The fixed-path operators have no numeric
            # comparison, and putting the threshold in this module instead would bury the one
            # number in the finding that is a judgement rather than arithmetic.
            "saturation": ({"claim": {
                "gap_pp": card.get("indistinguishable_gap_pp"),
                "total": card.get("saturated_total"),
                "lowest_count": card.get("lowest_indistinguishable_count"),
            }} if card.get("saturated_total") else {}),
            # READ RATHER THAN ASSUMED ABSENT, since 2026-10-06. This was hardcoded to None under a
            # comment saying a card never says which rows a figure came from, which was true of
            # every card that existed when it was written and stopped being true the day
            # `senbonzakura report` began printing the partition. The consequence was not a missing
            # field: `a-rate-with-no-partition-beside-it` needs a rate with a denominator, so the
            # first card to publish BOTH its counts and its row set was the first card able to fire
            # a check saying it published neither. Still None for the cards that genuinely say
            # nothing, which is still part of the gap.
            "eval_split": card.get("stated_partition"),
            "limit": None,
            "source_path": doc.get("source_path"),
        }


__all__ = ["FORMAT_KEY", "FORMAT_VALUE", "ModelCardAdapter", "cardread"]
