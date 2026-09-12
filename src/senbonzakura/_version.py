# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The version, alone in a module that imports nothing.

It lived in `cli.py`, which imports torch, optuna and transformers, so asking this package what
version it was cost roughly four seconds and a gigabyte of resident memory. The entry point needs
it before it knows which command is being run, and `doctor` needs to be able to answer on a
machine where torch is exactly what is missing.
"""

# The `.devN` counts wheels BUILT FOR another machine, not commits.
#
#   .dev1  the end-to-end production test
#   .dev2  the first two-wheel cut. Built and verified, then superseded the same hour by the
#          Q-33 field rename, and never handed to anyone.
#   .dev3  the two-wheel cut carrying Q-32's NOTICE and Q-33's `token_text`. Built and its
#          hashes sent to the ROG peer, who held rather than running; superseded before use.
#   .dev4  cut for the ROG pass. Adds the float32 direction basis (which changes the edit
#          slightly), the snapshot pre-flight, and the model card.
#   .dev5  cut because dev4's SEARCH is steered by a 48-token budget while its result is
#          published at 192, so the optimiser selects for refusals that arrive after the cutoff.
#          One measured budget now, and a refusal below the floor. A dev5 run is NOT comparable
#          to dev4 or to the 2026-09-10 table: the search itself changed, not only the report.
#
# Two different wheels sharing a version is how a bug gets reported against a build nobody can
# identify, and pip's cache reuses the older one without saying so. Bump this, and
# `checker/src/senbonzakura_check/_version.py` with it, on every dev cut. Note that .dev2 was
# bumped past rather than reused even though it never left this machine: a version that was
# built is spent, because a wheel with that name existed and could have been copied.
# See RELEASING.md, "A dev-only cut".
__version__ = "0.4.0.dev5"
