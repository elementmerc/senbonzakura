#!/usr/bin/env bash
# Author:  Daniel Iwugo
# Comment: Christ is King
# Wire this project's own pre-commit gates into the shared baseline hook.
#
# Run once per clone. The baseline hook runs every executable in .githooks/local.d/ as a
# subprocess and blocks the commit on a non-zero exit; .githooks/ itself is excluded from
# version control here, so a fresh clone has the gate scripts but not the wiring, and this
# is what connects the two.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"

# TWO CLONES, TWO WIRINGS, ONE GATE.
#
# This used to exit 1 here, and that made the one control CONTRIBUTING.md promises impossible
# for anybody outside to install: `.githooks/` is excluded per clone, so a fresh clone never has
# it, so the installer always refused. The gate script existed, was tracked, was documented, and
# could not be switched on by the people the documentation was addressed to. Found by the
# Dual-Use Armourer, 2026-09-21.
#
# Publishing `.githooks/` to fix that would be worse than the gap. It is fleet-wide tooling
# carrying infrastructure identifiers and paths from another machine, and putting those in a
# public repository is the same class of mistake as the leak this gate exists to prevent.
#
# So a clone without it gets a standalone `pre-commit` that runs the SAME tracked gate script.
# Not a copy of the logic: the substance stays in one file and only the wiring differs, which is
# the arrangement the gate script's own header already describes.
if [ ! -d .githooks ]; then
    hooks=$(git rev-parse --git-path hooks)
    mkdir -p "$hooks"
    target="$hooks/pre-commit"

    # REFUSE RATHER THAN OVERWRITE. Somebody who already has a pre-commit hook has one for a
    # reason, and silently replacing it is how a tool that claims to add a safety control
    # removes one.
    if [ -e "$target" ] && ! grep -q "senbonzakura-local-gate" "$target" 2>/dev/null; then
        echo "$target already exists and was not written by this script." >&2
        echo "Read it, then add this line yourself if you want both:" >&2
        echo "    exec tools/hooks/precommit-prompt-artefacts.sh" >&2
        exit 1
    fi

    cat > "$target" <<'HOOK'
#!/usr/bin/env bash
# senbonzakura-local-gate
#
# Refuses a commit carrying retained prompts or generations. The substance is in the tracked
# script below; this file is only the wiring. Delete it to uninstall; nothing else was touched.
set -euo pipefail
exec tools/hooks/precommit-prompt-artefacts.sh
HOOK
    chmod +x "$target"

    echo "installed $target"
    echo "  it runs: tools/hooks/precommit-prompt-artefacts.sh"
    echo "  delete that file to uninstall. Nothing else was changed."
    echo
    echo "This is not everything CI runs. It is the one control that has to fire BEFORE a"
    echo "commit, because a prompt or a generation that reaches history is not removed by"
    echo "deleting it afterwards."
    exit 0
fi

mkdir -p .githooks/local.d
ln -sfn ../../tools/hooks/precommit-prompt-artefacts.sh .githooks/local.d/10-prompt-artefacts.sh

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
