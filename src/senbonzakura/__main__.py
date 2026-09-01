# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
import sys

from .entry import main

if __name__ == "__main__":
    # `sys.exit(main())` rather than a bare `main()`, so that `python -m senbonzakura` reports
    # what the console script reports. Commands that fail by RAISING SystemExit (argparse errors,
    # the loud refusals) propagated either way; commands that fail by RETURNING a status did not,
    # and the return value was discarded on the way out.
    #
    # `doctor` was the one that mattered: it exists to be run before a long job, it printed
    # "this install cannot do what it claims" over nine failed checks, and it exited 0. Anything
    # using it as a pre-flight gate in a script or a CI job got a silent pass. `interactive`'s
    # non-terminal refusal and `headtohead_report`'s failure code were losing the same way.
    sys.exit(main())
