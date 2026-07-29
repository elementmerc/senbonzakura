#!/usr/bin/env bash
# for-hephaestus-check.sh — the SessionStart half of the planning gate.
#
# One file per project, `private/for-hephaestus.md`, says what the project is
# for and what it owes. The morning brief and the Sunday review both read it, so
# a project without one is invisible to planning no matter how much work lands
# in it.
#
# Why a SessionStart gate rather than a fleet-wide sweep: the declarations in
# that file (goals, horizon, status) are the OPERATOR's to make, not a tool's. A
# sweep can only infer them, and an inferred priority nobody chose is a
# manufactured expectation that rots into guilt. But when the operator opens the
# project, they are RIGHT THERE. The agent can ask, and write down the answer.
# That turns the one thing a tool must not do into the one thing the seam is
# perfect for.
#
# Three modes:
#   - missing:  MANDATORY. Interview the operator and create the file first.
#   - seeded:   the file exists but its goals are still the registry blurb.
#   - stale:    owned, but nobody has verified it in a while.
#
# Never blocks. Fail-open by design: any hiccup prints nothing and exits 0.
# Deliberately reads NOTHING as shell. This hook's sibling `panel-check.sh` used
# to `.` the project config, which executed whatever a repo-tracked file said,
# on session start (fixed 2026-07-26). Nothing here is evaluated.
set -u

ROOT="${CLAUDE_PROJECT_DIR:-$PWD}"
cd "$ROOT" 2>/dev/null || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

PLAN="private/for-hephaestus.md"
say() { echo "[planning] $*"; }

# Only speak in a repo with enough history to be a project rather than a scratch
# clone or a vendored reference.
#
# A SHALLOW clone breaks that test: `git rev-list --count` reports the fetched
# depth, not the real history, so a shallow clone of a large upstream looks
# identical to a one-commit scratch directory and gets skipped in silence.
# Observed on contributions/malchela (depth 1). Shallowness implies a real
# upstream worth cloning, so treat it as a project and let the later checks
# decide.
commits=$(git rev-list --count HEAD 2>/dev/null) || exit 0
if [ ! -f "$(git rev-parse --git-dir 2>/dev/null)/shallow" ]; then
    [ "${commits:-0}" -ge 5 ] || exit 0
fi

# A vendored reference checkout is not ours to plan. The convention is that ours
# have a private tree; a reference repo never does.
if [ ! -e private ] && [ ! -f "$PLAN" ]; then
    case "$ROOT" in
        */reference-repos/*|*/contributions/*) exit 0 ;;
    esac
fi

# ── Mode 1: no plan at all ──────────────────────────────────────────────────
if [ ! -f "$PLAN" ]; then
    say "MANDATORY: this project has no private/for-hephaestus.md."
    say ""
    say "It is the single file Hephaestus reads to know what this project is for,"
    say "so without it this repo is invisible to the morning brief and to the"
    say "Sunday review no matter how much work lands here."
    say ""
    say "Before other substantive work this session, the agent MUST:"
    say "  1. read this codebase enough to describe it honestly;"
    say "  2. ASK THE OPERATOR, do not infer, the DECLARATIONS below;"
    say "  3. write private/for-hephaestus.md in exactly the shape below;"
    say "  4. before writing, make sure private/ exists and is git-ignored:"
    say "       mkdir -p ~/sync/projects-private/<slug> && ln -s that private"
    say "       then ignore the literal word 'private' with NO trailing slash"
    say "       (a trailing slash matches directories and private/ is a symlink)"
    say "       in .gitignore if this repo has no public remote, or in"
    say "       .git/info/exclude if it does, since naming a private tree in a"
    say "       public .gitignore advertises that it exists."
    say ""
    say "THE EXACT SHAPE. YAML frontmatter between --- lines, then prose:"
    say ""
    say "  project: <slug>                 # DECLARATION"
    say "  status: active                  # active | dormant | ad-hoc | archived"
    say "  phase: <one line on where this actually is>"
    say "  updated: YYYY-MM-DD             # OBSERVATION: content last CHANGED"
    say "  checked: YYYY-MM-DD             # OBSERVATION: currency last VERIFIED"
    say "  until: YYYY-MM-DD               # ad-hoc ONLY; the time-box end"
    say "  next_6_months:                  # DECLARATION. May be empty, never invented."
    say "    - text: <what>"
    say "      severity: high              # high | medium | low"
    say "      date: YYYY-MM-DD            # optional; a past date is flagged for pruning"
    say "  queue:                          # DECLARATION. ~10 bare one-line strings."
    say "    - <a thing worth doing next>"
    say "  done:                           # OBSERVATION"
    say "    - text: <what got finished>"
    say "      date: YYYY-MM-DD"
    say ""
    say "  then '# Goals' (why this exists, the operator's words, not a task list)"
    say "  and '# History' (append-only, written at session close on a real delta)."
    say ""
    say "THE RULE THAT GOVERNS IT. Two kinds of field, and they are not equal:"
    say "  OBSERVATIONS (updated, checked, done) are facts. Git says the commits"
    say "  landed. Fill these in yourself, no permission needed."
    say "  DECLARATIONS (status, phase, next_6_months, queue, Goals) are the"
    say "  operator's intent. ASK. Never infer them, never 'reasonably assume'."
    say "  These fields rank the morning brief, so inventing one sets a priority"
    say "  the operator never chose and then reports it back as though they did."
    say ""
    say "Two dates, not one, and the difference is load-bearing: a deliberately"
    say "dormant project keeps 'checked' fresh while 'updated' ages, so it reads"
    say "as quiet-and-correct rather than rotten. One field cannot say that."
    say ""
    say "Empty is a valid answer. An empty next_6_months is honest; a horizon"
    say "invented to fill the field is worse than no horizon at all."
    exit 0
fi

# ── Read the frontmatter WITHOUT executing anything ─────────────────────────
val() { sed -n "s/^$1:[[:space:]]*//p" "$PLAN" 2>/dev/null | head -1 | tr -d '"'"'"''; }
checked=$(val checked)
status=$(val status)

# ── Mode 2: seeded but not yet owned ────────────────────────────────────────
if grep -q '_Seeded from portfolio.toml' "$PLAN" 2>/dev/null; then
    say "this project's plan is SEEDED but not yet yours."
    say "Its goal is still the registry blurb and its horizon is empty."
    say "Ask the operator what this project is really for and rewrite the Goals"
    say "section in their words, then delete the seeded marker line."
    exit 0
fi

# ── Mode 3: owned, but is it current ────────────────────────────────────────
if [ -n "$checked" ]; then
    # Pure date arithmetic; no shell evaluation of file content.
    then_s=$(date -d "$checked" +%s 2>/dev/null) || exit 0
    now_s=$(date +%s)
    days=$(( (now_s - then_s) / 86400 ))
    if [ "$days" -ge 21 ]; then
        say "plan last verified ${days}d ago (status: ${status:-unknown})."
        say "Worth a look at the queue and horizon while you are in here."
    fi
fi
exit 0
