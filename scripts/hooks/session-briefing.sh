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

# 2b. Multi-persona review gate: NOT called here. It runs as its own SessionStart
# hook entry in .claude/settings.json instead.
#
# It used to run in both places, so every session printed the panel line twice
# (measured 2026-07-30). Removing the copy from THIS file rather than from
# settings.json is deliberate: this briefing's output is large enough that the
# harness spills it to a file and shows only a preview, so a gate buried in here
# can go unread. A standalone hook entry arrives on its own. Anything that must
# be seen belongs in settings.json, not in this bundle.

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

# 4. Do the docs still describe the tree? Counts maintained by hand drift, and
# always downward: a file that says how many of something a project has is wrong
# the moment one is added, and nothing breaks, which is the cost. A session
# orienting from that file does not know what it was not told about.
#
# THE EXAMPLES THAT MOTIVATED THIS ARE DELIBERATELY NOT QUOTED HERE. This file
# is shared machinery installed into every repository on the fleet, and it is
# TRACKED in several of them, some with public remotes. A comment naming one
# project's crate count, ADR numbering and trait boundaries is FALSE about every
# other repository it lands in, and committing it publishes one project's
# internals as a description of another. Found 2026-09-17 by the senbonzakura
# peer, who diffed an uncommitted copy rather than assuming it was a stray edit:
# eight repositories carried it uncommitted, four of them with GitHub remotes.
#
# The detail lives with the project it is true of. Silent when clean.
if [ -x "$HERE/check-doc-drift.py" ] && command -v python3 >/dev/null 2>&1; then
    adr_out="$(python3 "$HERE/check-doc-drift.py" 2>&1)"; adr_rc=$?
    if [ "$adr_rc" -ne 0 ]; then
        echo "[doc gate] ${adr_out}"
        echo "[doc gate] Fix the named file; the check says which count is wrong and what it should be."
    fi
fi

# The seam map and the dispatch classification, both added 2026-09-07.
#
# Both are the constraint version of a rule that gets written down and then
# forgotten: a document can describe a boundary that does not exist yet, and a
# run can go around the orchestrator to a host nothing is watching. Each check is
# skipped in a repository that has no such script, which is most of them.
#
# The incidents behind both are recorded with the project they happened to, for
# the reason given at check 4: this file is installed everywhere and tracked in
# several places, so a comment true of one repository is a false claim about the
# rest.
#
# Silent when clean, like the checks above. A gate nobody runs is not a gate,
# which is why they are wired here rather than left as scripts someone could
# choose to invoke.
if [ -f "$ROOT/scripts/gen-seam-graph.py" ] && command -v python3 >/dev/null 2>&1; then
    seam_out="$(cd "$ROOT" && python3 scripts/gen-seam-graph.py --check 2>&1)"; seam_rc=$?
    if [ "$seam_rc" -ne 0 ]; then
        echo "[seam gate] ${seam_out}"
    fi
fi

if [ -f "$ROOT/scripts/check-dispatch-paths.py" ] && command -v python3 >/dev/null 2>&1; then
    disp_out="$(cd "$ROOT" && python3 scripts/check-dispatch-paths.py 2>&1)"; disp_rc=$?
    if [ "$disp_rc" -ne 0 ]; then
        echo "[dispatch gate] ${disp_out}"
    fi
fi

# The walk test: can a session opening this repo cold reach what it needs?
# Silent when every route resolves.
if [ -f "$ROOT/scripts/walk-test.py" ] && command -v python3 >/dev/null 2>&1; then
    walk_out="$(cd "$ROOT" && python3 scripts/walk-test.py --quiet 2>&1)"; walk_rc=$?
    if [ -n "$walk_out" ]; then
        printf '%s\n' "$walk_out"
    fi
fi

# The claims gate, at session start as well as at push.
#
# It is wired into pre-push, and pre-push is a poor adjudication point wherever
# the working rhythm accumulates many commits between pushes: for the whole of a
# work cluster the claims gate, and everything else on the push ladder, is
# dormant. It has been found red for dozens of commits while a manual sweep that
# believed it had fixed the same drift missed that the automated gate was
# already holding the answer in its hand.
#
# Stated without the project's own numbers, per check 4: this file is installed
# fleet-wide and tracked in several repositories, so a specific incident quoted
# here is a false claim about every repository it is not true of.
#
# It costs milliseconds and it moves adjudication from "when we push" to "when
# we start", which is when there is someone here to act on it. It overlaps the
# doc gate above on one claim and carries three the doc gate does not.
if [ -f "$ROOT/.githooks/claims-gate.py" ] && command -v python3 >/dev/null 2>&1; then
    claims_out="$(cd "$ROOT" && python3 .githooks/claims-gate.py 2>&1)"; claims_rc=$?
    if [ "$claims_rc" -ne 0 ]; then
        echo "[claims gate] ${claims_out}"
        echo "[claims gate] A document no longer describes the tree. Fix the named file."
    fi
fi

# ROUND TRIPS PER TOOL CALL, from the last session that did any work.
#
# Every request re-reads the whole conversation, so the bill is set by how many
# requests there are, and a request happens per assistant TURN rather than per
# tool call. The harness already runs every call in a response before asking
# again; what decides the number is whether a turn issues one call or four.
#
# Measured 2026-09-15 on one session: 1.03 calls per round trip, 97% of
# tool-calling turns issuing exactly one, against a median request of 418,549
# tokens. So a turn that could have batched and did not costs about 400k
# cache-read tokens. Nothing had ever looked, which is section 21's whole
# argument: a rule nothing measures is a rule that quietly stops being followed.
#
# Reported rather than enforced, deliberately. Plenty of calls genuinely depend
# on the previous result and a gate here would be wrong most times it fired.
# Takes about 3 seconds on an 800MB transcript.
# HEPHAESTUS ONLY, AND THE REASON IS SCOPE RATHER THAN OWNERSHIP.
#
# Measured 2026-09-17 after a peer reported two dead briefing blocks in cascade:
# 13 of the fleet's 14 repos had BOTH of these silently doing nothing, because
# the call sites travel in this shared template while the scripts travel in a
# separate list inside install-project.sh, and only one of the three was ever on
# that list. One list got the callers, another got the callees, and nothing
# diffed them.
#
# Porting the scripts was the obvious fix and the wrong one. Neither of these
# asks a question ABOUT THE REPOSITORY IT RUNS IN:
#
#   round-trips.py                     reads this MACHINE's transcripts
#   check-for-hephaestus-freshness.sh  sweeps the WHOLE factory
#
# Thirteen copies would print thirteen identical answers and then drift. So the
# blocks stay in one place and say nothing elsewhere, which is correct rather
# than merely quiet: in another repo there is genuinely nothing repo-specific
# for them to report.
#
# Contrast check-private-tree-traps.sh below, which IS invoked --repo and does
# answer differently in every tree. That one is distributed, and rightly.
#
# This is deliberately NOT the "print not installed here" fix. That is the right
# answer for a check that is MEANT to be everywhere and is missing, and it stays
# available for the next one; it is the wrong answer for a check that is
# hephaestus-only by design, because it would print two permanent noise lines in
# 13 repos every session, and a briefing people skim is a briefing that stops
# working.
_is_hephaestus() { [ -f "$ROOT/docs/decisions/all.md" ] && [ -d "$ROOT/crates/aegis-core" ]; }

if _is_hephaestus && [ -x "$ROOT/scripts/round-trips.py" ] && command -v python3 >/dev/null 2>&1; then
    rt="$(cd "$ROOT" && timeout 20 ./scripts/round-trips.py --json 2>/dev/null)"
    if [ -n "$rt" ]; then
        echo "$rt" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
r, share = d.get("calls_per_round_trip"), d.get("single_call_share")
med, turns = d.get("median_request_tokens"), d.get("turns_with_a_tool_call")
if r is None:
    sys.exit(0)
print(f"[round trips] last session: {r} tool call(s) per request over {turns:,} requests", end="")
if share is not None:
    print(f", {share:.0%} of them single-call", end="")
print(".")
if med:
    print(f"[round trips] median request {med:,} tokens, so each avoidable round trip costs about that.")
if share is not None and share > 0.8:
    print("[round trips] Batch independent calls into one turn where they do not depend on each other.")
'
    fi
fi

# HAS EACH PROJECT TOLD THE PLANNER WHAT IT IS DOING?
#
# chronos reads every project's `private/for-hephaestus.md` when it picks the
# week, so a file that has stopped being written is a planner reading an old
# picture with no way to know. Measured 2026-09-15: nine of seventeen active
# projects were behind, hephaestus itself by 417 commits.
#
# Reported here rather than gated, because the fix is the operator writing a
# paragraph and a gate cannot do that for him. Only the projects that are BEHIND
# are printed: the ones that are current are not news.
if _is_hephaestus && [ -x "$ROOT/scripts/check-for-hephaestus-freshness.sh" ]; then
    fh=$(timeout 60 "$ROOT/scripts/check-for-hephaestus-freshness.sh" 2>&1 \
         | grep -E "commit\(s\) since for-hephaestus|has no for-hephaestus|are behind" | head -8)
    if [ -n "$fh" ]; then
        echo "[planner] projects whose work is not reaching the week picker:"
        printf '%s\n' "$fh" | sed 's/^/          /'
    fi
fi

# ── Private-tree traps ──────────────────────────────────────────────────────
#
# TOLD BEFORE YOU TOUCH IT, which is the entire value. On 2026-09-16 a session
# in Stegcore ran `git merge origin/dev` and then `git merge --abort`, and 101
# files under private/ were deleted: origin/dev tracked them and HEAD ignored
# them, so the merge wrote over the live ones (git clobbers an IGNORED file
# without complaint) and the abort removed the lot. Recovered from restic.
#
# The session found the trap by walking into it. Nothing in the repo said the
# two branches meant different things, and `git status` inside private/ reported
# clean throughout, because a broken .git stub there sent git up to the parent.
#
# Scoped to THIS repo and run last, because it is the only check here that reads
# every remote ref. Silent when there is nothing to say, per principle 17: this
# must be distinguishable from not having run.
if [ -x "$ROOT/scripts/check-private-tree-traps.sh" ]; then
    traps=$(timeout 45 "$ROOT/scripts/check-private-tree-traps.sh" --repo "$ROOT" 2>/dev/null \
            | grep -v '^✓' | head -14)
    if [ -n "$traps" ]; then
        echo "[private-tree] this repo has a shape that has already destroyed files once:"
        printf '%s\n' "$traps" | sed 's/^/  /'
    fi
fi

echo "──────────────────────────────────────────────────────────────────"
exit 0
