# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Read the claims out of a model card, and work out what its own numbers can support.

WHY THIS IS THE WIDEST-REACH READER IN THE PACKAGE

The three original adapters read evaluation-harness result files. Hardly any model publisher
publishes one. What they publish is a card: a Markdown README with the numbers written into prose
and tables. So the artefact that reaches the most publishers is the one nothing could read, and
this closes that.

THE DISCIPLINE, AND IT IS THE SAME REFUSAL AS EVERYWHERE ELSE IN THIS PACKAGE

A card is unstructured text, so a reader of one can be wrong in a way a reader of JSON cannot. That
makes over-reach the real risk: a parser that mistakes a date for a score produces a confident
wrong audit **aimed at a stranger's published work**, which is worse than reading nothing. Every
extraction rule here is therefore deliberately narrow and refuses when unsure:

- a fraction is a score only when it is plainly one, and `10/1/2025` is rejected because a date is
  not a score. The guard is a refusal to match a slash-separated run of three or more numbers.
- a percentage is a score only inside 0 to 100.
- a total outside 2 to 1,000,000 is not a sample size.
- nothing is inferred from a claim's wording about which benchmark it belongs to. The surrounding
  line is carried verbatim so a reader can judge, rather than the parser guessing.

WHAT IT CAN SAY THAT IS WORTH SAYING

The strongest finding available from a card alone is arithmetic rather than opinion: **a saturated
score on a small sample cannot be told apart from a materially worse one.** For the worked example
that prompted this, a published claim of "zero intelligence loss" evidenced by 50 out of 50:

    a perfect 50/50 has a 95% Wilson interval of [92.9%, 100%]
    43/50 = 86.0% has [73.8%, 93.1%], which still overlaps it
    42/50 = 84.0% has [71.5%, 91.7%], which does not

So 43/50 is the boundary, and the honest sentence is that the card's own evidence cannot
distinguish a perfect model from one that lost seven answers in fifty, which is one in seven. That
is not a criticism of the edit. It is a statement about the test, and the publisher is usually the
person it helps most.
"""
from __future__ import annotations

import re

#: A total outside this range is not a sample size. Below 2 there is nothing to be a proportion of,
#: and above a million the "fraction" is almost certainly something else entirely.
MIN_TOTAL, MAX_TOTAL = 2, 1_000_000

#: Confidence level, as the multiplier. 1.96 is the 95% two-sided normal quantile, which is what
#: the main package's `metrics.wilson_interval` uses, so a figure checked here and a figure
#: measured there are on the same instrument rather than two intervals wearing one name.
Z = 1.96

#: A fraction that is plausibly a score. The lookarounds are the guard that keeps a date out: a
#: slash-separated run of three numbers, `10/1/2025`, must not match its leading `10/1`.
_FRACTION = re.compile(r"(?<![\d./])(\d{1,7})\s*/\s*(\d{1,7})(?![\d./])")

#: A percentage. Bounded to 0 to 100 after matching, because `150%` is rhetoric rather than a rate.
_PERCENT = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*%")

#: Phrases that assert no capability was lost. Matched case-insensitively on the whole card.
#:
#: DELIBERATELY A CLOSED LIST OF STRONG CLAIMS, not a sentiment detector. Each one asserts an
#: absolute, which is what makes it checkable: "zero loss" is a claim a number can fail to support,
#: whereas "performs well" is not a claim at all and nothing here should pretend to audit it.
ABSOLUTE_CLAIMS = (
    "zero intelligence loss", "zero capability loss", "zero capability degradation",
    "zero degradation", "zero performance loss", "zero quality loss", "zero loss",
    "no intelligence loss", "no capability loss", "no capability degradation",
    "no degradation", "no performance loss", "no quality loss",
    "no measurable loss", "no measurable degradation",
    "lossless", "fully preserved", "completely preserved", "perfectly preserved",
    "identical performance", "unchanged capability", "unchanged performance",
    "without any loss", "without capability loss", "without degradation",
)

#: YAML frontmatter keys worth lifting. A closed list for the same reason the GGUF reader has one:
#: the question is narrow and a card's frontmatter can carry a great deal that has no bearing on it.
FRONTMATTER_KEYS = ("license", "base_model", "library_name", "pipeline_tag", "language")

#: Words that mean the line is COUNTING something rather than SCORING it.
#:
#: FOUND BY A FALSE POSITIVE ON A REAL CARD, which is why this list exists and why it is a
#: refusal rather than a weighting. `wangzhang/gemma-4-31B-it-abliterated` carries the line
#: "| Optimization trials completed | 60/60 |", and a saturated-score check read that as a perfect
#: score on a sample of 60 and would have told the publisher their evidence was too thin. It is not
#: evidence at all; it is a progress counter, and the publisher's actual scores on that same card
#: are reported carefully and non-saturated.
#:
#: That card is worth reading as the counter-example to the whole check: it reports "7/100 refusals"
#: against a baseline of "99/100", shows three trials rather than one, and says in its own words
#: "We report 7/100 refusals honestly", while criticising other publishers for claiming near-perfect
#: scores. A check that fired on it would have been attacking the one publisher already doing the
#: right thing.
COUNTING_WORDS = (
    "trial", "trials", "step", "steps", "epoch", "epochs", "layer", "layers",
    "shard", "shards", "file", "files", "commit", "commits", "seed", "seeds",
    "run", "runs", "checkpoint", "checkpoints", "iteration", "iterations",
    "completed", "progress", "done", "batch", "batches", "token", "tokens",
    "parameter", "parameters", "tensor", "tensors",
)


#: Labels a card uses to say which rows a figure was taken on. A CLOSED LIST, for the same reason
#: `FRONTMATTER_KEYS` is closed: the question is narrow, and a line this misread would assert a
#: partition the publisher never stated, which is worse than reporting the gap.
#:
#: WHY THIS EXISTS AT ALL, 2026-10-06. `a-rate-with-no-partition-beside-it` fires on a rate whose
#: artefact names no row set, and the model-card adapter answered that question with a hardcoded
#: "none", under a comment saying a card never says which rows a figure came from. That premise
#: stopped being true the day `senbonzakura report` began printing the partition, and the first
#: card to carry both a denominator and the rows it came from was reported as carrying neither. A
#: checker that fires on the one card doing the right thing is the failure this package's own
#: `COUNTING_WORDS` list exists because of, one layer along.
PARTITION_LABELS = (
    "which rows of it",
    "evaluation split",
    "eval split",
    "partition",
    "split",
)

#: Values that are a statement of absence rather than the name of a row set. Matched after the
#: markdown is stripped, so `**not recorded in this artefact**` is read as the gap it is instead of
#: becoming a partition called "not recorded in this artefact".
_NO_PARTITION = ("not recorded", "not stated", "not known", "unknown", "unmeasured", "n/a",
                 "none", "-", "")

# Where a label has to sit for its value to be the card's own statement: at the start of a line,
# after any list, quote or table punctuation, optionally emphasised. Anchored, so a sentence
# EXPLAINING what the row set field means is prose about the field rather than the field, and is
# not read as one.
def _partition_pattern(label: str) -> re.Pattern[str]:
    return re.compile(r"^[\s>*|_-]*`?" + re.escape(label) + r"`?(?:\*\*)?\s*[:|]\s*(.+)$",
                      re.IGNORECASE | re.MULTILINE)


def stated_partition(text: str) -> str | None:
    """Which rows the card says its figures were taken on, or None when it does not say.

    Labels are tried in `PARTITION_LABELS` order rather than in document order, so an explicit
    wording wins over a bare `split:` somewhere else on the page. None is returned for a label
    whose value is itself a statement of absence, because reading "not recorded" as the name of a
    partition would turn the gap into a reassurance.
    """
    for label in PARTITION_LABELS:
        match = _partition_pattern(label).search(text)
        if not match:
            continue
        value = match.group(1).strip().strip("|").strip().strip("*`").strip()
        if value.lower() not in _NO_PARTITION and not any(
                p in value.lower() for p in ("not recorded", "not stated", "unmeasured")):
            return value
    return None


def wilson_interval(count: int, n: int, z: float = Z) -> tuple[float, float]:
    """A confidence interval for a proportion that behaves at small n.

    Transcribed from `senbonzakura.metrics.wilson_interval` rather than imported, for the reason
    `gguf_read` sets out at length: this distribution declares no dependencies, so the main package
    is not on the path for anybody who installed only the checker. There is a test asserting the two
    agree to twelve decimal places, so the copy cannot drift silently.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = count / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def lowest_indistinguishable(count: int, n: int) -> int | None:
    """The smallest score out of `n` whose interval still overlaps `count`'s, or None.

    "Overlaps" is the honest test for "a reader cannot tell these apart from this evidence". None
    when there is no such score below `count`, which happens when the sample is large enough that
    even one fewer correct answer separates cleanly.

    Walks down from `count` rather than solving for the boundary, because the walk is obviously
    right at a glance and `n` is bounded at a million.
    """
    if n <= 0 or count < 0 or count > n:
        return None
    target_low, _ = wilson_interval(count, n)
    lowest = None
    for c in range(count - 1, -1, -1):
        _, hi = wilson_interval(c, n)
        if hi >= target_low:
            lowest = c
        else:
            break
    return lowest


def _frontmatter(text: str) -> dict:
    """The YAML frontmatter's wanted keys, read without a YAML parser.

    No dependency is available and the keys wanted are all simple scalars or short lists, so this
    reads `key: value` at the top level and stops. A key whose value is a nested structure is
    skipped rather than half-read: a partly-parsed value is worse than an absent one.
    """
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    out = {}
    for line in text[3:end].splitlines():
        if not line or line.startswith((" ", "\t", "#", "-")):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key, value = key.strip(), value.strip().strip("\"'")
        if key in FRONTMATTER_KEYS and value:
            out[key] = value
    return out


def _context(text: str, start: int, end: int) -> str:
    """The line a match sits on, collapsed to one line and trimmed.

    Carried verbatim so a reader can judge what the number refers to. The parser deliberately does
    not try to work that out: a guessed benchmark name attached to somebody else's figure is the
    kind of confident wrongness this module exists not to produce.
    """
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line = text[line_start: line_end if line_end != -1 else len(text)]
    return " ".join(line.split())[:240]


def looks_like_a_count(context: str) -> bool:
    """Whether this line is counting something rather than scoring it.

    Word-boundary matching, so `runs` matches and `overruns` does not, and `file` does not fire on
    `profile`. See `COUNTING_WORDS` for the real card that forced this.
    """
    words = set(re.findall(r"[a-z]+", context.lower()))
    return bool(words & set(COUNTING_WORDS))


def extract_claims(text: str) -> list[dict]:
    """Every score claim the card states plainly, as dicts.

    A claim carries `kind` (`fraction` or `percent`), its numbers, whether it is saturated, and the
    line it came from. Order is the order they appear, so a reader can find them.

    Deduplicated on the whole claim, because a table row holding the same figure twice yields two
    identical matches from one line and a claim count inflated by formatting is not a count of
    claims. The real example: a card's total row reading `| **Total** | **50/50 (100%)** |
    **50/50 (100%)** |` produced four identical entries.

    ONE FIGURE WRITTEN TWICE IS ONE CLAIM, which is the same rule applied across the two forms.
    `0/128 = 0.0%` is a fraction and a percentage of the same thing, and reading it as two claims
    produced a second entry with no denominator from a line that plainly carries one. That is not
    only a count being inflated: `null-reported-as-zero` fires on a zero with no sample size
    beside it, so a card that started stating its denominators became the first card able to fire
    a check about not stating them. The percentage is dropped and the fraction kept, because the
    fraction is the form that carries the sample size.
    """
    claims = []
    seen = set()
    # Fraction values already claimed on each line, so a percentage restating one can be
    # recognised. Keyed on the line text, which is what `_context` returns.
    fractions_by_line: dict[str, list[float]] = {}
    for m in _FRACTION.finditer(text):
        count, total = int(m.group(1)), int(m.group(2))
        if not (MIN_TOTAL <= total <= MAX_TOTAL) or count > total:
            continue
        context = _context(text, m.start(), m.end())
        claim = {
            "kind": "fraction",
            "count": count,
            "total": total,
            "value": count / total,
            # A COUNTER IS NOT A SCORE, so it can never be a saturated score however complete it
            # is. Recorded as a claim anyway, because dropping it would hide from a reader that
            # the parser saw the line at all.
            "saturated": count == total and not looks_like_a_count(context),
            "is_count": looks_like_a_count(context),
            "context": context,
        }
        key = (claim["kind"], count, total, context)
        if key in seen:
            continue
        seen.add(key)
        claims.append(claim)
        fractions_by_line.setdefault(context, []).append(claim["value"])
    for m in _PERCENT.finditer(text):
        value = float(m.group(1))
        if not (0.0 <= value <= 100.0):
            continue
        context = _context(text, m.start(), m.end())
        key = ("percent", value, context)
        if key in seen:
            continue
        # The same figure already read in the form that carries its sample size. Compared at the
        # precision the percentage is printed to, so `3/128 = 2.3%` matches and a genuinely
        # different percentage on the same line does not.
        if any(round(f * 100.0, 1) == round(value, 1) for f in fractions_by_line.get(context, ())):
            continue
        seen.add(key)
        claims.append({
            "kind": "percent",
            # NO DENOMINATOR, TRUTHFULLY. A percentage on its own does not carry one, and
            # inventing one in order to compute an interval is the fabrication this package
            # exists to catch in other people's work.
            "count": None,
            "total": None,
            "value": value / 100.0,
            "saturated": value == 100.0 and not looks_like_a_count(context),
            "is_count": looks_like_a_count(context),
            "context": context,
        })
    return claims


def find_absolute_claims(text: str) -> list[str]:
    """The absolute no-loss claims the card makes, in the words it used."""
    low = text.lower()
    return [phrase for phrase in ABSOLUTE_CLAIMS if phrase in low]


def read_card(text: str, *, name: str | None = None) -> dict:
    """A card's claims and what they can support, as a plain dict."""
    claims = extract_claims(text)
    absolutes = find_absolute_claims(text)

    # The strongest saturated claim that carries a denominator, which is the one worth reporting:
    # the largest sample is the publisher's best evidence, so judging the card on it is the
    # generous reading rather than picking their weakest line.
    saturated = [c for c in claims if c["saturated"] and c["total"]]
    best = max(saturated, key=lambda c: c["total"]) if saturated else None

    worst = lowest_indistinguishable(best["count"], best["total"]) if best else None
    low = high = None
    if best:
        low, high = wilson_interval(best["count"], best["total"])

    return {
        "card_name": name,
        "frontmatter": _frontmatter(text),
        # Which rows the card says its figures came from, or None. Read rather than assumed: see
        # `PARTITION_LABELS` for the premise that stopped being true.
        "stated_partition": stated_partition(text),
        "claims": claims,
        "claim_count": len(claims),
        "absolute_claims": absolutes,
        "absolute_claim_count": len(absolutes),
        # The saturated claim under examination, or None.
        "saturated_total": best["total"] if best else None,
        "saturated_context": best["context"] if best else None,
        "saturated_interval_low": low,
        "saturated_interval_high": high,
        # The lowest score that this evidence cannot tell apart from the perfect one, as a count
        # and as a share. None when nothing overlaps, which means the sample was big enough.
        "lowest_indistinguishable_count": worst,
        "lowest_indistinguishable_share": (worst / best["total"]) if (worst and best) else None,
        # THE SIZE OF THE DOUBT, IN PERCENTAGE POINTS, and this is the number the check reads.
        #
        # Measured before the check was written, because the obvious version of it was wrong: a
        # perfect score overlaps something lower at EVERY sample size, so "does anything overlap"
        # is universal and firing on it would be cry-wolf. What varies is how far down the overlap
        # reaches, and that is a real property of the evidence:
        #
        #     n=50   -> cannot be told from 86.0%, a gap of 14.0 points
        #     n=100  -> 93.0%, 7.0 points
        #     n=200  -> 96.5%, 3.5 points
        #     n=1000 -> 99.3%, 0.7 points
        "indistinguishable_gap_pp": (
            (1.0 - worst / best["total"]) * 100.0 if (worst is not None and best) else None),
        # Does the card assert an absolute while offering no number at all to support it?
        "absolute_claim_without_any_number": bool(absolutes) and not claims,
    }


__all__ = [
    "ABSOLUTE_CLAIMS", "COUNTING_WORDS", "FRONTMATTER_KEYS", "MAX_TOTAL", "MIN_TOTAL",
    "PARTITION_LABELS", "Z",
    "extract_claims", "find_absolute_claims", "looks_like_a_count",
    "lowest_indistinguishable", "read_card", "stated_partition",
    "wilson_interval",
]
