#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The corpus builder, reachable from a checkout with nothing installed.

The builder itself is `senbonzakura.corporabuild`, and the command is `senbonzakura corpora`. It
moved into the package because `tools/` ships in no wheel, so the one command an install-from-git
user has to run was the one command their install did not contain.

This file stays because the build pipeline runs before anything is installed: a `src/` layout
means `python -m senbonzakura.corporabuild` needs an install, and the release workflow builds the
corpora in order to build the wheel that would provide it. It adds `src` to the path and calls
the real thing. Nothing else lives here, so the two cannot drift.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from senbonzakura.corporabuild import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
