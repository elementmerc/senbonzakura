#!/usr/bin/env bash
# session-briefing.sh — the single SessionStart briefing.
#
# One opening block so nothing promised, unpushed, or drifting hides across
# sessions. Aggregates three fail-open checks in a fixed order:
#   1. unpushed / uncommitted work (session-push-check.sh, all branches)
#   2. the deferred-work ledger      (deferred-check.sh, baseline Section 24)
#   3. fleet drill drift + a due disaster drill (~/.hephaestus/drills markers)
#
# Every section is independently fail-open: a failure in one prints nothing and
# never blocks the session. Consolidates workflow-gates Fix 3 (push) and Fix 7
# (drill cadence) with the existing deferred gate.
set -u

ROOT="${CLAUDE_PROJECT_DIR:-$PWD}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRILLS="${HOME:-$ROOT}/.hephaestus/drills"

echo "── session briefing ──────────────────────────────────────────────"

# 1. Unpushed / uncommitted work (across all local branches).
if [ -x "$HERE/session-push-check.sh" ]; then
    PUSH_CHECK_ALL_BRANCHES=1 CLAUDE_PROJECT_DIR="$ROOT" bash "$HERE/session-push-check.sh" || true
fi

# 2. Deferred-work ledger.
if [ -x "$HERE/deferred-check.sh" ]; then
    CLAUDE_PROJECT_DIR="$ROOT" bash "$HERE/deferred-check.sh" || true
fi

# 3. Fleet drill drift + disaster-drill-due markers.
if [ -d "$DRILLS" ]; then
    drift_found=0
    for m in "$DRILLS"/DRIFT-* "$DRILLS"/DISASTER-DRILL-DUE; do
        [ -e "$m" ] || continue
        drift_found=1
        echo "[drill gate] $(basename "$m"): $(head -1 "$m" 2>/dev/null)"
    done
    if [ "$drift_found" -eq 1 ]; then
        echo "[drill gate] A drill flagged drift or is due. Investigate, then clear the marker:"
        echo "[drill gate]   rm ${DRILLS}/DRIFT-* ${DRILLS}/DISASTER-DRILL-DUE   (once resolved)"
    fi
fi

# Multi-persona review gate (baseline Section 25): the MANDATORY adoption
# bootstrap in a repo that has not adopted it, else commits since the last
# panel review. Wired by install-project.sh; fail-open, never blocks.
if [ -x "$HERE/panel-check.sh" ]; then
    CLAUDE_PROJECT_DIR="$ROOT" bash "$HERE/panel-check.sh" || true
fi

echo "──────────────────────────────────────────────────────────────────"
exit 0
