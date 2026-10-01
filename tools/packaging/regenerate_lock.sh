#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King
#
# Regenerate requirements.lock from constraints.txt.
#
# WHY THIS IS A SCRIPT AND NOT A LINE IN A COMMENT
#
# The CI job that verifies the lockfile's hashes goes red whenever a pin moves and the lock was
# not regenerated in the same commit. That is correct behaviour, and it is only fair if
# regenerating is a command somebody can run rather than a recipe they have to reconstruct. The
# lockfile's own header carried the recipe and nothing checked that the header stayed true, so
# the day `uv` changed a flag or the pinned Python moved, the job would have gone red for a
# reason nobody could fix from the file.
#
# `tests/test_lock_regeneration_is_runnable.py` asserts that the command below and the command
# the lockfile's header documents are the same command. Change one and the test names the other.
#
# --no-deps IS LOAD-BEARING, and it is the whole design of the two-file split. Without it the
# resolver adds every transitive dependency at whatever version is current today, which is a
# different environment wearing this filename. With it, the twelve versions are exactly the
# twelve in constraints.txt and all that is added is what they hash to.

set -euo pipefail

# THE COMMAND. One definition, read by the test. Keep the flags on this line.
UV_COMPILE_COMMAND="uv pip compile --generate-hashes --no-deps --python-version 3.14 constraints.txt"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [ "${1:-}" = "--print-command" ]; then
    # For the test, so it reads this file's definition rather than a copy of it.
    printf '%s\n' "$UV_COMPILE_COMMAND"
    exit 0
fi

# Pre-flight, rather than failing halfway through and leaving a half-written lock.
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not installed, and pip's own compile does not produce this format." >&2
    echo "       Install it from https://docs.astral.sh/uv/ then re-run this script." >&2
    exit 1
fi
if [ ! -f constraints.txt ]; then
    echo "ERROR: constraints.txt is missing, and it is the input this lock is compiled from." >&2
    exit 1
fi

echo "==> regenerating requirements.lock from constraints.txt"
echo "    $UV_COMPILE_COMMAND"

# Written to a temporary file and moved into place, so an interrupted run cannot leave a
# truncated lockfile behind. A truncated lock is the one failure mode the CI job's count
# assertion exists to catch, and leaving one behind here would be this script causing it.
tmp="$(mktemp "${TMPDIR:-/tmp}/requirements.lock.XXXXXX")"
trap 'rm -f "$tmp"' EXIT

# shellcheck disable=SC2086
# Word splitting is intended: UV_COMPILE_COMMAND is a command line, defined in this file, and
# the test asserts it matches the lockfile header. Quoting it would pass the whole string as one
# argument.
if ! $UV_COMPILE_COMMAND --output-file "$tmp"; then
    echo "ERROR: uv pip compile failed. requirements.lock is unchanged." >&2
    exit 1
fi

pins="$(grep -cE '^[a-zA-Z][a-zA-Z0-9._-]*==' "$tmp" || true)"
declared="$(grep -cE '^[a-zA-Z][a-zA-Z0-9._-]*==' constraints.txt || true)"
if [ "$pins" -ne "$declared" ]; then
    echo "ERROR: the new lock names ${pins} pins and constraints.txt names ${declared}." >&2
    echo "       Refusing to install it. A lock with a different pin count than its input is" >&2
    echo "       either a truncated compile or a --no-deps flag that stopped applying." >&2
    exit 1
fi
if ! grep -q 'sha256:' "$tmp"; then
    echo "ERROR: the new lock carries no hashes, so --generate-hashes did not apply." >&2
    echo "       Refusing to install it: a lock without hashes is the thing this file is not." >&2
    exit 1
fi

# The generated body replaces the body and the hand-written header is kept. uv writes its own
# short preamble; this file's header carries the reasoning, the two install commands and the
# regeneration recipe, and none of that survives a compile.
header_end="$(grep -nE '^[a-zA-Z][a-zA-Z0-9._-]*==' requirements.lock | head -1 | cut -d: -f1)"
if [ -z "$header_end" ]; then
    echo "ERROR: could not find where the header ends in the existing requirements.lock." >&2
    exit 1
fi

{
    head -n "$((header_end - 1))" requirements.lock
    grep -vE '^#' "$tmp" | sed '/^$/d'
} > "${tmp}.merged"
mv "${tmp}.merged" requirements.lock

echo "==> requirements.lock regenerated: ${pins} pins, $(grep -c 'sha256:' requirements.lock) hashes"
echo "    Now run the verification the CI job runs:"
echo "      pip download --require-hashes --no-deps -r requirements.lock -d \$(mktemp -d)"
