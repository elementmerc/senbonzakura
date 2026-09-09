# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Reading TOML on every Python this project claims to support.

`tomllib` entered the standard library in 3.11. `pyproject.toml` declares
`requires-python = ">=3.10"`, and four test files imported `tomllib` unconditionally, so on the
DECLARED MINIMUM interpreter the suite could not even be collected: CI's 3.10 job failed at import
with "No module named 'tomllib'", and had done since the tests were written.

Nobody saw it because every machine anyone ran the suite on was newer. This project claims 3.10 in
its metadata and on a badge, and 3.10 was the one version where the claim was false.

`tomli` is the same parser; `tomllib` was adopted from it. It is a test-only dependency, declared
in the `dev` extra for `python_version < "3.11"` alone, so nothing a user installs grows because
of it.
"""
try:
    import tomllib
except ModuleNotFoundError:   # Python 3.10
    import tomli as tomllib

load = tomllib.load
loads = tomllib.loads

__all__ = ["load", "loads", "tomllib"]
