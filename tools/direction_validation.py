#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Thin shim: the validation experiments now live in the package as `senbonzakura validate`.

Kept because run specs on the measurement machines invoke this path, and a spec that has already
been launched should not break because the code moved. New work should use the subcommand.
"""
import sys

from senbonzakura.validate import main

if __name__ == "__main__":
    sys.exit(main())
