# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`senbonzakura check`, which lives in the `senbonzakura-check` distribution.

WHY THIS SHIM EXISTS

Decision Q-29 ships the checker as its own distribution so that `pip install senbonzakura-check`
costs seconds and pulls nothing, while `pip install senbonzakura` keeps meaning "I want to
abliterate a model" (Q-27). Two distributions cannot both own files inside one import package,
so the checker owns `senbonzakura_check` and this package depends on it.

That leaves `senbonzakura check` needing somewhere to dispatch to, and this is it. The direction
matters: the big package imports the small one, never the other way round. Nothing under
`senbonzakura_check` may import `senbonzakura`, and a test walks its import graph to keep that
true, because an upward import would silently put torch back into the torch-free distribution
and would only show up on somebody else's machine.

Both spellings reach the same code. `senbonzakura check` is here for people who already have the
full install; `senbonzakura-check` is the console script the small distribution provides.
"""
from senbonzakura_check.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    # THE GUARD THIS SHIM IS REQUIRED TO CARRY, and the first version did not.
    #
    # `tests/test_exit_status.py` exists because eight modules once had no `__main__` guard, so
    # `python -m senbonzakura.<module>` executed nothing and exited 0 while the documentation
    # invoked them that way. Without this line the shim reintroduced that exact defect for
    # `check`: the package form exited 2 on a missing argument and the module form exited 0
    # having run nothing, and the two disagreed about whether anything had happened.
    raise SystemExit(main())
