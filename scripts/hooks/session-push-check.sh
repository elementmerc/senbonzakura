#!/usr/bin/env bash
# Session push gate: surface unpushed commits so work does not sit unpushed at
# session close (three-month review: 17 of ~40 handoffs opened with "dev N
# ahead, UNPUSHED"). Wired as a Claude Code SessionStart hook (highly visible:
# output lands in the opening context) and, best-effort, as a SessionEnd hook.
# Also callable by hand or from /handoff.
#
# Reports, for the current branch: how many commits are ahead of the tracked
# upstream (or, if there is no upstream, how many commits are on no remote),
# plus whether the tree is dirty. Never pushes anything; never blocks.
#
# Fail-open by design: any problem prints nothing useful and exits 0. A hook
# that gates a session on a git hiccup would be worse than the problem it warns
# about.
set -u

ROOT="${CLAUDE_PROJECT_DIR:-$PWD}"
cd "$ROOT" 2>/dev/null || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) || exit 0
[ "$branch" = "HEAD" ] && exit 0   # detached; nothing meaningful to nudge

ahead=0
tracked=1
if upstream=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null); then
    ahead=$(git rev-list --count "${upstream}..HEAD" 2>/dev/null) || ahead=0
else
    tracked=0
    # No upstream: count commits on this branch that are on no remote at all.
    ahead=$(git rev-list --count HEAD --not --remotes 2>/dev/null) || ahead=0
fi

dirty=0
[ -n "$(git status --porcelain 2>/dev/null)" ] && dirty=1

# Nothing to say when everything is pushed and clean.
if [ "${ahead:-0}" -eq 0 ] && [ "$dirty" -eq 0 ]; then
    exit 0
fi

if [ "${ahead:-0}" -gt 0 ]; then
    if [ "$tracked" -eq 1 ]; then
        echo "[push gate] ${ahead} commit(s) on '${branch}' not pushed to ${upstream}:"
    else
        echo "[push gate] ${ahead} commit(s) on '${branch}' are on no remote ('${branch}' has no upstream):"
    fi
    git log --oneline -n 5 "HEAD~$(( ahead < 5 ? ahead : 5 ))..HEAD" 2>/dev/null | sed 's/^/    /'
    if [ "$tracked" -eq 1 ]; then
        echo "    push with: git push"
    else
        echo "    set an upstream and push: git push -u origin ${branch}"
    fi
fi

if [ "$dirty" -eq 1 ]; then
    echo "[push gate] working tree on '${branch}' has uncommitted changes (git status)."
fi

# Optional: report OTHER local branches that are ahead of their upstream. The
# session-briefing sets PUSH_CHECK_ALL_BRANCHES=1 so nothing unpushed hides on a
# branch you are not currently standing on.
if [ "${PUSH_CHECK_ALL_BRANCHES:-0}" = "1" ]; then
    while IFS= read -r b; do
        [ -z "$b" ] && continue
        [ "$b" = "$branch" ] && continue
        up=$(git rev-parse --abbrev-ref --symbolic-full-name "${b}@{upstream}" 2>/dev/null) || continue
        n=$(git rev-list --count "${up}..${b}" 2>/dev/null) || continue
        [ "${n:-0}" -gt 0 ] && echo "[push gate] other branch '${b}' is ${n} ahead of ${up}."
    done < <(git for-each-ref --format='%(refname:short)' refs/heads 2>/dev/null)
fi

echo "[push gate] Don't leave the session with work unpushed unless you mean to."
exit 0
