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
