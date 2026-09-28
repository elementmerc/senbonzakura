#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The entry point `.pre-commit-hooks.yaml` runs, so the hook installs nothing at all.

WHY THIS FILE EXISTS

pre-commit's `language: python` runs `python -mpip install .` in a clone of the hook's repository,
read from pre-commit 4.6.2's own `languages/python.py`. This repository's root distribution is
`senbonzakura`, whose dependencies are torch, transformers, accelerate, optuna, pyarrow and
sentencepiece, and there is no way to tell pre-commit to install a subdirectory instead. So the
hook used to put roughly a gigabyte into somebody's commit hook, while the manifest beside it
claimed the environment was "small and fast to build" on the grounds that the checker imports
nothing beyond the standard library. That reasoned about what the code IMPORTS when the question
was what packaging RESOLVES, which is the same mistake `action.yml` records.

`language: script` resolves the entry inside the hook repository and installs NOTHING
(`unsupported_script.py`: `install_environment = lang_base.no_install`). The names `script` and
`system` are still accepted and map to those modules in `clientlib.py`. Since the checker has
`dependencies = []` and imports only the standard library, an isolated environment buys nothing
here: there is no dependency to isolate. So the hook now runs this file straight out of the clone
pre-commit already made, at whatever `rev` the user pinned.

WHY NOT A MIRROR REPOSITORY, which is the usual answer to this shape of problem: it would be a
second public repository to tag in step with this one, and a second place for the version to drift.
This costs one file and keeps one home.

WINDOWS IS COVERED. pre-commit parses the shebang itself rather than relying on the OS, in
`parse_shebang.normalize_cmd`, whose docstring says so in as many words.
"""
import pathlib
import sys

#: The floor `checker/pyproject.toml` declares. pip would enforce it on an ordinary install and
#: cannot here, because there is no install: this runs on whatever `python3` the user's PATH finds.
#: Said plainly rather than left to a SyntaxError or an obscure AttributeError further in.
MINIMUM_PYTHON = (3, 10)

if sys.version_info < MINIMUM_PYTHON:
    got = ".".join(str(n) for n in sys.version_info[:3])
    want = ".".join(str(n) for n in MINIMUM_PYTHON)
    sys.stderr.write(
        f"senbonzakura-check needs Python {want} or newer and this hook is running on {got} "
        f"({sys.executable}).\n"
        f"  pre-commit runs this with the interpreter on your PATH, because the hook deliberately "
        f"installs nothing.\n"
        f"  Set `language_version` on the hook in your .pre-commit-config.yaml, or put a newer "
        f"python3 first on PATH.\n")
    raise SystemExit(1)

# The package is not installed, by design, so it is imported from the tree pre-commit cloned.
# Resolved from THIS file rather than from the working directory, which is the repository being
# committed to and has nothing to do with where the checker lives.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from senbonzakura_check.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
