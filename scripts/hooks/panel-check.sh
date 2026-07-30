#!/usr/bin/env bash
# panel-check.sh — the soft, per-cluster half of the multi-persona review gate
# (baseline Section 25), plus the first-run adoption bootstrap.
#
# Two modes, chosen by whether this repo has adopted the gate (the single signal
# is PANEL_GATE_ENABLED in .baseline-hook-config):
#   - NOT adopted: print a MANDATORY bootstrap block. The agent's next move, before
#     other work, is to propose a roster and enable the gate.
#   - adopted: count commits since the last panel artefact and nudge when a
#     feature cluster looks reviewable.
#
# Never blocks. Fail-open by design: any git or filesystem hiccup prints nothing
# and exits 0.
#
# The HARD half is .githooks/promotion-gate.sh, called from pre-push, which
# refuses a dev to main promotion whose commits no panel artefact covers. It
# landed 2026-07-30, having been described as existing since ADR 63. Note that
# it does NOT read PANEL_GATE_ENABLED: this hook may be configured, a gate may
# not (ADR 67).
set -u

ROOT="${CLAUDE_PROJECT_DIR:-$PWD}"
cd "$ROOT" 2>/dev/null || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

# Adoption signal. Sourced from .baseline-hook-config (same file the git hooks
# read); unset or 0 means this repo has not adopted the gate yet.
# Read the ONE value this hook needs, without executing the file.
#
# This used to `. ./.baseline-hook-config`, which runs whatever that file
# contains. The git hooks stopped doing that on 2026-07-25 (ADR 66) but this one
# was missed, and it is the worse of the two: a git hook fires on commit or push,
# whereas a SessionStart hook fires when the operator merely OPENS the project.
# A repo-tracked config carrying a shell command therefore executed on open, in
# 11 repos.
#
# The pattern accepts only `PANEL_GATE_ENABLED=0` or `=1`, optionally quoted.
# Anything else leaves the default. There is nothing here for a config to
# inject into, because nothing is evaluated.
PANEL_GATE_ENABLED=0
if [ -f .baseline-hook-config ]; then
    _pg=$(sed -n 's/^[[:space:]]*PANEL_GATE_ENABLED[[:space:]]*=[[:space:]]*["'"'"']\{0,1\}\([01]\)["'"'"']\{0,1\}[[:space:]]*$/\1/p' \
          .baseline-hook-config 2>/dev/null | head -1)
    case "$_pg" in 0|1) PANEL_GATE_ENABLED="$_pg" ;; esac
fi

# ── Bootstrap mode: not adopted ─────────────────────────────────────────────
if [ "$PANEL_GATE_ENABLED" != "1" ]; then
    # Only speak in a repo with enough history to be worth reviewing.
    commits=$(git rev-list --count HEAD 2>/dev/null) || exit 0
    [ "${commits:-0}" -ge 3 ] || exit 0
    echo "[panel gate] MANDATORY (baseline Section 25): this repo has NOT adopted the"
    echo "[panel gate] multi-persona review gate. Before any other substantive work this"
    echo "[panel gate] session, the agent MUST:"
    echo "[panel gate]   1. look through this codebase to understand what it is;"
    echo "[panel gate]   2. offer the operator at least 5 candidate review personas, each a"
    echo "[panel gate]      distinct lens (Marvel/DC and real-life options both welcome);"
    echo "[panel gate]   3. once a roster is agreed, record it in CLAUDE.md and set"
    echo "[panel gate]      PANEL_GATE_ENABLED=1 in .baseline-hook-config;"
    echo "[panel gate]   4. only then proceed. Tell the operator plainly this is the"
    echo "[panel gate]      instructed move, so they can act without having to remember."
    echo "[panel gate] When the panel later runs, every persona is a SEPARATE SUBAGENT with"
    echo "[panel gate] its own context, launched in parallel, never shown another's output."
    echo "[panel gate] Overlap between them is the measurement, not waste."
    exit 0
fi

# ── Nudge mode: adopted ─────────────────────────────────────────────────────
# The latest panel artefact and the commit it recorded covering.
last=$(ls -1t private/reviews/*panel*.md 2>/dev/null | head -1 || true)

if [ -z "$last" ]; then
    commits=$(git rev-list --count HEAD 2>/dev/null) || exit 0
    if [ "${commits:-0}" -ge 5 ]; then
        echo "[panel gate] gate adopted but no panel review recorded yet. Run /panel to review a feature cluster (baseline Section 25)."
    fi
    exit 0
fi

# A gate that cannot measure must SAY SO, never fall quiet. Silence here is
# indistinguishable from "nothing to report", so every unmeasurable case below
# speaks. This was not academic: every adopted repo had panel artefacts with no
# `covers:` line, so the gate exited silently at 61 and had never once spoken
# since adoption. Absence and malfunction must not share a representation.
base=$(grep -oiE 'covers:[[:space:]]*[0-9a-f]{7,40}' "$last" 2>/dev/null \
       | grep -oiE '[0-9a-f]{7,40}' | head -1 || true)

if [ -z "$base" ]; then
    echo "[panel gate] CANNOT MEASURE: $(basename "$last") has no \`covers: <sha>\` line,"
    echo "[panel gate] so there is no way to tell what it reviewed or what has landed since."
    echo "[panel gate] Add a line reading  covers: <sha>  to that artefact (the commit the"
    echo "[panel gate] panel read up to). Until then this gate cannot report drift."
    exit 0
fi

if ! git cat-file -e "${base}^{commit}" 2>/dev/null; then
    echo "[panel gate] CANNOT MEASURE: $(basename "$last") claims to cover ${base}, which is"
    echo "[panel gate] not a commit in this repo. A rebase or a wrong sha; correct the artefact."
    exit 0
fi

since=$(git rev-list --count "${base}..HEAD" 2>/dev/null) || exit 0
when=$(basename "$last" | grep -oE '^[0-9]{4}-[0-9]{2}-[0-9]{2}' || echo "?")
if [ "${since:-0}" -ge 1 ]; then
    echo "[panel gate] ${since} commit(s) since the last panel review (${when}). Run /panel when the cluster is ready; a dev-to-main promotion needs a panel artefact covering it."
else
    echo "[panel gate] up to date: panel review ${when} covers HEAD."
fi
exit 0
