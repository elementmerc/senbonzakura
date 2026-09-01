# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The version, alone in a module that imports nothing.

It lived in `cli.py`, which imports torch, optuna and transformers, so asking this package what
version it was cost roughly four seconds and a gigabyte of resident memory. The entry point needs
it before it knows which command is being run, and `doctor` needs to be able to answer on a
machine where torch is exactly what is missing.
"""

__version__ = "0.3.0"
