# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The reporting floor can be read without loading the code `metrics.py` copies from Heretic.

Measured in a fresh interpreter, because inside the test session `metrics` is long since imported
and `sys.modules` would say so whatever `floor` does.
"""
import subprocess
import sys

from senbonzakura import floor, metrics


def test_the_floor_and_metrics_state_the_same_number():
    # Stated twice on purpose (metrics imports nothing); this is what stops the two drifting.
    assert floor.MIN_REPORTABLE_N == metrics.MIN_REPORTABLE_N == 30


def test_importing_the_floor_does_not_load_metrics():
    probe = ("import sys; from senbonzakura.floor import MIN_REPORTABLE_N; "
             "print('senbonzakura.metrics' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         timeout=60, check=True)
    assert out.stdout.strip() == "False"
