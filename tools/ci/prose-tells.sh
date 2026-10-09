# GENERATED FILE. Do not edit by hand.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King
#
# The tell patterns from the project's pre-commit hook, exported so the CI checker has a pattern
# list to read: the hooks are kept out of this repository deliberately, so a public checkout has
# no hook to parse. The hook remains the source of truth, this is a copy of it, and a test fails
# if the two drift.
#
# Regenerate with:  python tools/dev/export_prose_tells.py
#
# The syntax below is the hook's bash array, kept verbatim so one parser reads both.

LLM_TELL_PATTERNS=(
        # Negative parallelism — strong AI signature.
        "(it'?s|this[[:space:]]+is|that'?s)[[:space:]]+not[[:space:]]+(just|only|merely|simply|about)[[:space:]]+[A-Za-z]"
        "not[[:space:]]+just[[:space:]]+[A-Za-z]+[[:space:]]+(but|it'?s|they'?re|we'?re)"
        # THE CONTRACTED SPELLING, AND AN ARBITRARY SUBJECT. Added 2026-10-06 after a
        # negative parallelism went through a hand review of an outreach draft AND through a
        # purpose-written scanner, because both looked for the literal "not just" and the
        # sentence said "isn't just": "the number isn't just a number, it's a number your run
        # was large enough to support". The two patterns above catch 2 of 6 real examples.
        # They miss every contraction (isn't / aren't / doesn't / wasn't), which matters
        # because §14 of the baseline mandates contractions BY DEFAULT, so this gate was
        # blind to exactly the spelling the house style prefers. They also miss an
        # uncontracted parallelism whose subject is a noun rather than it/this/that, as in
        # "the metric is not just a count, it's a claim".
        #
        # Both patterns below require the SECOND clause (comma, then a pronoun), which is
        # what makes the shape a parallelism rather than an ordinary negation. That is what
        # keeps "we don't just measure refusal rate here" and "don't just take my word for
        # it, check the source" from firing. Verified against 6 tells and 5 legitimate uses:
        # 6 caught, 0 false positives.
        "n'?t[[:space:]]+(just|only|merely|simply)[[:space:]]+[^,;.!?]{1,60},[[:space:]]*(it|its|it'?s|they|they'?re|that|this|we|we'?re|you)\b"
        "[[:space:]]+(is|are|was|were)[[:space:]]+not[[:space:]]+(just|only|merely|simply)[[:space:]]+[^,;.!?]{1,60},[[:space:]]*(it|its|it'?s|they|they'?re|that|this|we|you)\b"
        # Hedge / weight phrases.
        "it'?s[[:space:]]+(important|worth|critical|essential)[[:space:]]+to[[:space:]]+note"
        "it[[:space:]]+should[[:space:]]+be[[:space:]]+noted"
        "needless[[:space:]]+to[[:space:]]+say"
        # Sycophantic openers (mostly chat, but they leak into docs).
        "^[[:space:]]*(Great[[:space:]]+question|Absolutely|Certainly)!"
        # The LLM vocabulary watchlist.
        '\b(delve|delves|delving|leverage|leverages|leveraged|leveraging|tapestry|realm|holistic|paradigm|multifaceted|seamless|seamlessly|robust|robustness|unleash|unleashes|unleashing|elevate|elevates|elevating|foster|fosters|fostering|comprehensive|comprehensively|intricately|profoundly|navigate(s|d|)[[:space:]]+(the[[:space:]]+complexities|the[[:space:]]+landscape|the[[:space:]]+realm))\b'
        # Curly / smart quotes (signal AI rendering or autocorrect).
        $'“|”|‘|’'
        # Paragraph-start transition crutches.
        "^(Moreover|Furthermore|Additionally|In[[:space:]]+conclusion|In[[:space:]]+summary)[[:space:]]*,"
    )
