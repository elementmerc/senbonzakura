#!/usr/bin/env bash
# Project pre-commit gate: refuse a commit that carries retained prompts or generations.
#
# Wired in through the baseline hook's project extension point, which runs every
# executable in .githooks/local.d/ as a subprocess and blocks on a non-zero exit:
#
#     ln -s ../../tools/precommit-prompt-artefacts.sh .githooks/local.d/10-prompt-artefacts.sh
#
# The symlink lives there because .githooks/ is excluded per-clone (the shared baseline
# installs it from outside this repository), so the substance has to be here, tracked and
# reviewable, with only the wiring outside. `tools/install-local-hooks.sh` makes the link.
#
# CI runs the same checker over the tree. This is the earlier of the two gates: CI catches
# it after the commit exists, and by then the prompts are in the history.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"

for py in python3 python; do
    if command -v "$py" >/dev/null 2>&1; then
        exec "$py" tools/check_prompt_artefacts.py --staged
    fi
done

# No interpreter is not a pass. A gate that cannot run has to say so and stop, because the
# alternative is a commit that was never checked reporting itself as clean.
echo "prompt-artefact gate: no python interpreter found, so nothing could be checked" >&2
exit 2
