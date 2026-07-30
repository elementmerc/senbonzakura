#!/usr/bin/env bash
# Wire this project's own pre-commit gates into the shared baseline hook.
#
# Run once per clone. The baseline hook runs every executable in .githooks/local.d/ as a
# subprocess and blocks the commit on a non-zero exit; .githooks/ itself is excluded from
# version control here, so a fresh clone has the gate scripts but not the wiring, and this
# is what connects the two.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"

if [ ! -d .githooks ]; then
    echo "no .githooks/ in this clone: install the shared baseline hooks first" >&2
    exit 1
fi

mkdir -p .githooks/local.d
ln -sfn ../../tools/precommit-prompt-artefacts.sh .githooks/local.d/10-prompt-artefacts.sh

echo "wired:"
for gate in .githooks/local.d/*; do
    printf '  %s -> %s\n' "$gate" "$(readlink "$gate" 2>/dev/null || echo "(not a link)")"
done

hooks_path=$(git config core.hooksPath || true)
if [ "$hooks_path" != ".githooks" ]; then
    echo
    echo "note: core.hooksPath is '${hooks_path:-unset}', not .githooks, so these will not run." >&2
    echo "      run: git config core.hooksPath .githooks" >&2
    exit 1
fi
