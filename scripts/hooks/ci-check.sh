#!/usr/bin/env bash
# ci-check.sh — surface the state of the remote build at session start.
#
# WHY THIS EXISTS, AND WHAT ITS ABSENCE COST
#
# CI was red on `dev` from 2026-08-03 to 2026-09-09. Thirty-seven days, eighteen
# consecutive failing runs, eight of ten jobs failing on every platform and every
# interpreter. Nothing told anyone. The SessionStart briefing surfaced unpushed
# work, the deferred ledger and the panel gate, and not this. The README carries a
# live CI badge, so the public repository and the PyPI page rendered a failing
# build for five weeks while six review passes and an eight-persona panel each ran
# the suite locally, on the one machine that has the untracked build artefacts, and
# reported it green.
#
# The cause was mundane: tests that need artefacts built at release time were not
# guarded to skip on a checkout, so they failed everywhere except the author's box.
# The lesson is not about those tests. It is that a check whose answer nobody reads
# is not a check, and CI was the loudest instrument in the project.
#
# Never blocks, and never slows a session down: no gh, no auth, no network, no
# output. Fail-open in every direction, like the other hooks here.
set -u

ROOT="${CLAUDE_PROJECT_DIR:-$PWD}"
cd "$ROOT" 2>/dev/null || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

command -v gh >/dev/null 2>&1 || exit 0

# Every wait here is deadline-bounded, and macOS ships no `timeout`: it is GNU
# coreutils, available there as `gtimeout` only if somebody installed them. With
# neither, the choice is an unbounded network call in a SessionStart hook or no
# gate, and a hook that can hang the start of a session is the worse of the two.
if command -v timeout >/dev/null 2>&1; then
    bounded() { timeout "$@"; }
elif command -v gtimeout >/dev/null 2>&1; then
    bounded() { gtimeout "$@"; }
else
    exit 0
fi

# `gh auth status` is cheap and local. Without it every call below would prompt or
# hang, which is the one thing a SessionStart hook must never do.
bounded 5 gh auth status >/dev/null 2>&1 || exit 0

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) || exit 0
[ -n "$BRANCH" ] && [ "$BRANCH" != "HEAD" ] || exit 0

# One request, hard-bounded. A slow network must cost the session a few seconds at
# most, so a timeout is treated exactly like "no answer": silence.
RUNS=$(bounded 12 gh run list --branch "$BRANCH" --limit 12 \
        --json conclusion,status,headSha,createdAt,url 2>/dev/null) || exit 0
[ -n "$RUNS" ] && [ "$RUNS" != "[]" ] || exit 0

HEAD_SHA=$(git rev-parse HEAD 2>/dev/null) || HEAD_SHA=""

# The only completed run can be an OLD one. This hook reported "the last CI run
# FAILED" naming a commit two commits behind, six minutes after the build for HEAD
# had started and while it was still running; that build went on to pass all ten
# jobs. The verdict was true of the run it examined and false of the sentence it
# printed, which is this project's recurring failure shape rather than a new one.
# So the answer now carries WHICH commit it is about and whether anything newer is
# in flight, and the wording distinguishes the three.
LATEST=$(printf '%s' "$RUNS" | HEAD_SHA="$HEAD_SHA" python3 -c '
import json, os, sys
try:
    runs = json.load(sys.stdin)
except Exception:
    sys.exit(0)
first_done = next((i for i, r in enumerate(runs) if r.get("status") == "completed"), None)
if first_done is None:
    sys.exit(0)
done = runs[first_done:]
done = [r for r in done if r.get("status") == "completed"]
latest = runs[first_done]
# How far back the failure goes, so a five-week outage does not read like a blip.
streak = 0
for r in done:
    if r.get("conclusion") == "failure":
        streak += 1
    else:
        break
# "at least", when the streak fills the window we asked for. Saying "12 in a row"
# when 12 is all we looked at is a number about the query, not about the build.
capped = "1" if streak == len(runs) else "0"
# Runs newer than the one being reported on, still going. `gh run list` is newest
# first, so anything before `latest` in the list started after it.
newer = [r for r in runs[:first_done] if r.get("status") != "completed"]
head = (os.environ.get("HEAD_SHA") or "")
sha = (latest.get("headSha") or "")
# "-" for an absent field rather than an empty one: `read` with a whitespace IFS
# collapses runs of tabs, so an empty field silently shifts every field after it.
print("\t".join(v or "-" for v in [
    latest.get("conclusion") or "",
    str(streak),
    capped,
    sha[:7],
    (latest.get("createdAt") or "")[:10],
    latest.get("url") or "",
    ((newer[-1].get("headSha") or "")[:7] if newer else ""),
    ("1" if head and sha and head == sha else "0"),
    head[:7],
]))
') || exit 0
[ -n "$LATEST" ] || exit 0

IFS=$'\t' read -r CONCLUSION STREAK CAPPED SHA WHEN URL INFLIGHT IS_HEAD HEAD_SHORT <<< "$LATEST"
[ "$INFLIGHT" = "-" ] && INFLIGHT=""
[ "$HEAD_SHORT" = "-" ] && HEAD_SHORT="(unknown)"

if [ "$CONCLUSION" = "failure" ]; then
    if [ -n "$INFLIGHT" ]; then
        echo "[ci gate] The last COMPLETED CI run on '$BRANCH' failed ($SHA, $WHEN),"
        echo "[ci gate] but a newer run for $INFLIGHT is still going. Read that one before"
        echo "[ci gate] concluding anything about where the branch stands."
    elif [ "$IS_HEAD" = "1" ]; then
        echo "[ci gate] CI FAILED on the current HEAD ($SHA, $WHEN)."
    else
        echo "[ci gate] The last CI run on '$BRANCH' FAILED ($SHA, $WHEN), and HEAD"
        echo "[ci gate] ($HEAD_SHORT) has no completed run of its own."
    fi
    if [ "${STREAK:-0}" -gt 1 ]; then
        if [ "$CAPPED" = "1" ]; then
            echo "[ci gate] It has failed at least $STREAK runs in a row (all we looked at)."
        else
            echo "[ci gate] It has failed $STREAK runs in a row. This is not a blip."
        fi
    fi
    echo "[ci gate]   $URL"
    echo "[ci gate] A red build that nobody reads is how this project shipped 37 days"
    echo "[ci gate] of failing CI behind a green local suite. Look before you build on it."
elif [ "$CONCLUSION" != "success" ] && [ -n "$CONCLUSION" ]; then
    echo "[ci gate] The last CI run on '$BRANCH' ended '$CONCLUSION' ($SHA, $WHEN)."
fi
exit 0
