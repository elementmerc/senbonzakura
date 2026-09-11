# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The version, alone in a module that imports nothing.

It lived in `cli.py`, which imports torch, optuna and transformers, so asking this package what
version it was cost roughly four seconds and a gigabyte of resident memory. The entry point needs
it before it knows which command is being run, and `doctor` needs to be able to answer on a
machine where torch is exactly what is missing.
"""

# `.dev1` and not `.dev0` because a wheel was cut from it: the first cut handed to a machine this
# one cannot reach, for the end-to-end production test. Two different wheels sharing a version is
# how a bug gets reported against a build nobody can identify, and pip's cache reuses the older
# one without saying so. Bump this on every dev cut. See RELEASING.md, "A dev-only cut".
__version__ = "0.4.0.dev1"
