# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The checker's version, which is the same number as the abliterator's.

TWO DISTRIBUTIONS OUT OF ONE COMMIT MUST REPORT THE SAME VERSION, or a bug report against
"senbonzakura 0.4.0" cannot be traced to a build. They are separate files because this package
must not import `senbonzakura`: that direction is the one thing the split exists to prevent, and
reading the version across it would reintroduce it for the sake of one string.

Kept in step by `tests/test_second_distribution.py` rather than by anyone remembering.
"""

__version__ = "0.4.0.dev3"
