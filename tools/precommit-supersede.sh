#!/usr/bin/env bash
# A deleted tool must say what does its job now.
#
# THE RULE THIS GATES
#
#   A script is superseded when something else does its job, not when something else has a
#   similar name.
#
# Adopted 2026-08-17 after it caught two people in one afternoon, in both directions:
#
#   * `kl-after-gguf.sh` was read as "measures KL on a GGUF" and was about to justify vendoring
#     `llama-perplexity` into every wheel. It is eight lines of `pgrep` and `sleep` waiting for an
#     unrelated upload to stop competing for the broadband. "after-gguf" is temporal.
#   * `bench.sh` was on a delete list as superseded by a model benchmark. It is a Rust
#     build-and-test benchmark with nothing to do with models.
#   * `spec-bench.sh`, also on that list, held the only record of a wrong conclusion: a run
#     reported "closed, do not re-run" having loaded zero draft models, because a flag had been
#     renamed and the script silently ran with no speculation at all.
#
# Ten lines of reading catches each of those. This gate does not do the reading; nothing can. What
# it does is refuse to let the deletion pass silently, so the reading happens at the only moment
# somebody is definitely thinking about the file.
#
# WHAT IT ASKS FOR
#
# When a commit deletes an executable or a script under a tools directory, its message must say
# what replaced it, using one of:
#
#     Superseded-by: senbonzakura quantise
#     Superseded-by: nothing (it measured a job that no longer exists)
#
# "nothing" is a legitimate answer and is deliberately spelled out rather than allowed by silence.
# A script whose job is genuinely gone should be deleted, and saying so is one line.
#
# Fail-open on anything it cannot determine, per the baseline: a gate that misfires on a rebase is
# a gate somebody turns off.
set -u

_msg_file="${1:-}"
[ -n "$_msg_file" ] && [ -f "$_msg_file" ] || exit 0

# Staged deletions only. Renames are not deletions: git reports them as R and the file still exists.
_deleted=$(git diff --cached --name-only --diff-filter=D 2>/dev/null \
  | grep -E '\.(sh|py|bash|zsh|ps1)$' \
  | grep -E '(^|/)(tools|scripts|bin)/' || true)

[ -z "$_deleted" ] && exit 0

# A test file being deleted alongside the thing it tested is not a tool deletion.
_deleted=$(printf '%s\n' "$_deleted" | grep -vE '(^|/)tests?/' || true)
[ -z "$_deleted" ] && exit 0

if grep -qiE '^Superseded-by:[[:space:]]*\S' "$_msg_file"; then
    exit 0
fi

_count=$(printf '%s\n' "$_deleted" | grep -c . || true)
{
    echo "supersede gate: this commit deletes ${_count} script(s) and does not say what replaced them."
    echo
    printf '  %s\n' "$_deleted"
    echo
    echo "  A script is superseded when something else does its job, not when something else has a"
    echo "  similar name. Two files were nearly deleted on 2026-08-17 on the strength of their"
    echo "  names alone; one was a Rust build benchmark and one held the only record of a wrong"
    echo "  conclusion."
    echo
    echo "  Add a trailer naming the replacement, after reading the file:"
    echo
    echo "      Superseded-by: senbonzakura quantise"
    echo "      Superseded-by: nothing (its job no longer exists)"
    echo
    echo "  'nothing' is a real answer. It is spelled out rather than allowed by silence."
} >&2
exit 1
