#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The prompt-pool builder, reachable from a checkout with nothing installed.

The builder itself is `senbonzakura.trackbuild`, and the command is `senbonzakura track build`. It
moved into the package because `tools/` ships in no wheel, so the guide sent people who had just
installed to a script their install did not contain.

This file stays for the same reason its sibling does: a checkout with nothing installed. It adds
`src` to the path and calls the real thing, and holds nothing of its own, so the two cannot drift.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from senbonzakura.trackbuild import main

if __name__ == "__main__":
    sys.exit(main())
