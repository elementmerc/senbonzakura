# Third-party notices

Senbonzakura includes third-party code, which determines the licence of the
whole project.

## Heretic

`src/senbonzakura/metrics.py` contains a keyword-marker list (`HERETIC_MARKERS`)
and a normalisation function (`_heretic_norm`) copied verbatim from Heretic:

- Project: https://github.com/p-e-w/heretic
- Copyright (C) 2025-2026 Philipp Emanuel Weidmann and contributors
- Licence: AGPL-3.0-or-later

They are reproduced so that Senbonzakura's residual-refusal figures stay directly
comparable to Heretic's own keyword metric. Because this AGPL-licensed code is
included and distributed, Senbonzakura as a whole is distributed under
AGPL-3.0-or-later (see [LICENSE](LICENSE)).

### Statement of modification (AGPL-3.0 section 5(a))

**Senbonzakura is a modified work based in part on Heretic, and it is not
Heretic.** Section 5(a) of the licence requires a work like this one to say so
prominently and to give a date, so that nobody mistakes our behaviour for
upstream's.

- **Modified by:** Daniel Iwugo.
- **First included:** 2026-07-14, in the initial package.
- **Relicensed to AGPL-3.0-or-later for this inclusion:** 2026-07-17.
- **Most recent modification to the file carrying it:** 2026-09-07.

What was and was not changed, because the distinction is the whole point of the
notice:

- `HERETIC_MARKERS` and `_heretic_norm` are **byte-identical to upstream** and are
  deliberately kept that way. If they ever drift, the comparability that is the
  only reason for copying them is gone, and this notice becomes wrong.
- **Everything around them is ours**, including a stricter `is_refusal`, a soft
  refusal detector, and the surrounding module. Senbonzakura reports the Heretic
  keyword rate *alongside* its own metric rather than as its metric.
- **No behaviour of Heretic's is reimplemented.** Senbonzakura's search, its
  direction extraction and its measurement are independent work.

A reader comparing a Senbonzakura number to a Heretic number should know that
only the keyword rate is a like-for-like comparison, and that every other figure
is measured by our own instrument.
