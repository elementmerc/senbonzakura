#!/usr/bin/env bash
# SessionStart gate: surface the deferred-work ledger every session so promised
# "later" work gets re-evaluated instead of quietly evaporating into an old
# handoff. Output goes into the session's opening context (the agent reads it)
# and is visible to the operator.
#
# Fail-open by design: a problem here must never block a session, so every
# failure path just prints nothing and exits 0.
set -u

ROOT="${CLAUDE_PROJECT_DIR:-$PWD}"
LEDGER="$ROOT/DEFERRED.md"

if [[ ! -f "$LEDGER" ]]; then
    cat <<'MSG'
[deferred-work gate] This project has no DEFERRED.md yet.
Per the engineering baseline, build one at the start of this session: mine the
past session handoffs and transcripts, plus private/tech-debt.md,
private/decisions.md, and the "NEXT" sections of prior handoffs, for work that
was promised for "later" and never done. Record each as a `- [ ]` line in
DEFERRED.md, grouped by area. After that this gate surfaces them every session.
MSG
    exit 0
fi

open_count=$(grep -Ec '^[[:space:]]*- \[ \]' "$LEDGER" 2>/dev/null) || open_count=0

if [[ "$open_count" -eq 0 ]]; then
    echo "[deferred-work gate] DEFERRED.md has no open items. Add new deferrals here as they arise."
    exit 0
fi

echo "[deferred-work gate] ${open_count} open deferred item(s) in DEFERRED.md awaiting re-evaluation:"
grep -E '^[[:space:]]*- \[ \]' "$LEDGER" 2>/dev/null | sed -E 's/^[[:space:]]*- \[ \][[:space:]]*/  - /' | head -30
if [[ "$open_count" -gt 30 ]]; then
    echo "  ... and $((open_count - 30)) more (see DEFERRED.md)"
fi
echo "Re-evaluate with the operator: pick one up, re-scope it, or close it. Record any new deferrals here so they never hide in a handoff again."
exit 0
