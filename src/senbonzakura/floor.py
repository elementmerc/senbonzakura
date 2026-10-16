# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The smallest sample a rate may be reported over.

Kept apart from `metrics.py` on purpose. That module carries code copied from another project
under its own licence (see THIRD-PARTY-NOTICES.md), and anything that needs only this one number
should not have to load that code to get it. `metrics` states the same number itself, because it
imports nothing by design, and a test holds the two equal.

Three separate wrong numbers in one night, 2026-09-06, all the same shape: a rate computed over a
sample too small to carry it. "7.0% against 3.0%" was five observations against two and reversed
direction at n=1092. Five hashes of an empty string agreed with each other and were read as
determinism. A 280-row corpus collapsed to 40 distinct requests, and partitioned three ways would
have produced an AUC over about 13.
"""

#: Below this, a rate is not reported as a value at all. It is a floor against absurdity rather
#: than a guarantee of power: n=30 is still weak for comparing two rates, and the interval beside
#: the rate is what tells a reader how weak. Anything under it cannot support a claim in any
#: framing, so the honest output is a refusal rather than a figure with a caveat.
MIN_REPORTABLE_N = 30
