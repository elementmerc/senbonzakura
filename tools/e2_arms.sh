#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# e2_arms.sh — run one base model through every ablation method and measure what each one COST.
#
# WHY THIS EXISTS AS A FILE RATHER THAN A COMMAND IN A MESSAGE
#
# This experiment has been described in chat three times and has never existed anywhere a machine
# could read it. That is how an experiment gets lost, and separately it is how a GPU sits idle for
# hours: a heredoc pasted through an ssh chain ate its own quoting and the run never started. The
# standing rule since then is a shipped script, never an inline command. A person on the other
# machine runs one file and gets one directory back.
#
# WHAT IT ANSWERS
#
# Nobody in this field can currently say "for this base model, this method costs this much". The
# published claim is one to three percent degradation with no benchmark behind it; a comparative
# study puts one family at eight points on grade-school arithmetic. Neither side reports an
# interval. Every piece needed to do better already exists in this repository, and what is missing
# is the run.
#
# Each arm produces an abliterated model AND a capability score on the same fixed questions the
# stock model answered, so the number reported is a PAIRED change with an interval rather than two
# accuracies a reader has to subtract by eye.
#
# THE ARMS, AND WHY EACH ONE IS HERE
#
#   stock              no edit at all. The reference every other arm is compared against. Without
#                      it the other five produce absolute accuracies and answer nothing.
#   single-pass        one direction, full strength, no search. This is how the tools that retain
#                      capability best are described as working.
#   single-pass-raw    the same WITHOUT restoring row norms. A control, not a recommendation: it
#                      is the naive formulation, and it is here so the norm restoration has
#                      something to be better than.
#   searched           our default. Optimises strength, placement and direction count.
#   searched-one-      our search, one direction. The missing cell: single-pass differs from
#     direction        searched in TWO ways at once (no search AND one direction), so those two
#                      alone cannot say which half did the work, and direction count is this
#                      project's entire claim. This arm makes it a readable 2x2.
#   searched-rounds4   our default plus the leak fix. Restoring row lengths measurably undoes part
#                      of the ablation (5% to 46% depending on how uneven the lengths are); four
#                      alternating rounds remove the direction fully and keep the lengths. Whether
#                      a cleaner cut makes a BETTER model is exactly what this arm measures.
#   searched-capgate   our default with capability inside the search, so a config that wrecks
#                      reasoning is rejected during the search rather than reported after it. The
#                      only arm that PREVENTS damage rather than measuring it.
#
# ORDER IS DELIBERATE: cheapest first. A run stopped halfway still answers something.
#
# THE HELD-OUT BOUNDARY
#
# The capgate arm's in-search probe reads the HEAD of the eval set, so every arm is SCORED with
# --skip past it. Uniformly, including the arms whose search never touched the benchmark, because
# arms scored on different questions are not arms. This is not a nicety: three published numbers
# in this project's history described the rows they were selected on.
#
# USAGE
#
#   tools/e2_arms.sh --model Qwen/Qwen3-1.7B --track track --out runs/e2-qwen3-1.7b
#   tools/e2_arms.sh ... --arms stock,single-pass          # a subset
#   tools/e2_arms.sh ... --dry-run                         # print the plan, touch nothing
#   tools/e2_arms.sh ... --drop-weights                    # artefacts only, for a small disk
#   tools/e2_arms.sh ... --probe-n 40 --n 200              # holdout boundary and scored items
#   tools/e2_arms.sh ... --max-new 640                     # fix the budget, skipping the probe
#   tools/e2_arms.sh ... --no-budget-probe                 # same, keeping the default budget
#
# THE BUDGET IS SIZED BEFORE IT IS SPENT
#
# A run graded three arms at 20.5% of answers ungradeable, on every arm including the unedited
# reference, because the token budget was too small for a fifth of the benchmark. The answers that
# fail to finish are the LONG ones, so what gets graded is an easier exam than the one set. The
# stock model is probed first, doubling the budget until it clears, and every arm then uses what
# that found. One arm of generations to save six arms of unusable ones.
#
# Re-running is safe. Each arm writes a completion marker and a finished arm is skipped, so an
# interrupted run resumes rather than repeating GPU hours. Weights are KEPT unless you ask.
#
# The Hub token, if one is needed, is read from $HF_TOKEN by the tools themselves. It is never
# passed as an argument here, because arguments reach the process table and the logs.
#
set -euo pipefail

MODEL=""
TRACK="track"
OUT=""
EVAL_SET="openai/gsm8k:main::test"
TASK="numeric"
N=200
MAX_NEW=320
BUDGET_PROBE=1        # size MAX_NEW from the stock arm before spending it on six more
BATCH=8
TRIALS=60
PROBE_N=40
DEVICE="cuda"
# The default is every arm, so it doubles as the list an unknown-arm message quotes. One list,
# because a hand-kept second copy of a set of names is how the guard and the editor drifted apart
# in this codebase once already.
KNOWN_ARMS="stock,single-pass,single-pass-raw,searched,searched-one-direction,searched-rounds4,searched-capgate"
ARMS="$KNOWN_ARMS"
ARM_TIMEOUT=18000     # 5 hours, the operator's per-task ceiling on this machine
DRY_RUN=0
FORCE=0
DROP_WEIGHTS=0
SZ="senbonzakura"

die() { printf 'e2_arms: %s\n' "$*" >&2; exit 2; }

usage() {
  # The header block IS the documentation, so it is read from the file rather than restated here
  # where the two would drift. Stops at the first line that is not a comment.
  awk 'NR>2 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "$0"
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model)        MODEL="${2:-}"; shift 2 ;;
    --track)        TRACK="${2:-}"; shift 2 ;;
    --out)          OUT="${2:-}"; shift 2 ;;
    --eval)         EVAL_SET="${2:-}"; shift 2 ;;
    --task)         TASK="${2:-}"; shift 2 ;;
    --n)            N="${2:-}"; shift 2 ;;
    --max-new)      MAX_NEW="${2:-}"; BUDGET_PROBE=0; shift 2 ;;
    --no-budget-probe) BUDGET_PROBE=0; shift ;;
    --batch)        BATCH="${2:-}"; shift 2 ;;
    --trials)       TRIALS="${2:-}"; shift 2 ;;
    --probe-n)      PROBE_N="${2:-}"; shift 2 ;;
    --device)       DEVICE="${2:-}"; shift 2 ;;
    --arms)         ARMS="${2:-}"; shift 2 ;;
    --arm-timeout)  ARM_TIMEOUT="${2:-}"; shift 2 ;;
    --drop-weights) DROP_WEIGHTS=1; shift ;;
    --force)        FORCE=1; shift ;;
    --dry-run)      DRY_RUN=1; shift ;;
    -h|--help)      usage ;;
    *)              die "unknown option '$1'. Try --help." ;;
  esac
done

[ -n "$MODEL" ] || die "--model is required (e.g. Qwen/Qwen3-1.7B)."
[ -n "$OUT" ]   || die "--out is required: one directory holds the whole experiment."

case ",$ARMS," in
  *,stock,*) ;;
  *) die "the 'stock' arm is not in --arms. Every other arm is scored as a PAIRED change against
        it, so without it this run produces absolute accuracies and answers nothing. Add it, or
        point --out at a directory where a previous run already produced stock/capability.json." ;;
esac

# ---------------------------------------------------------------- pre-flight
# Everything checked before the first GPU hour, not halfway through the fourth.

command -v "$SZ" >/dev/null 2>&1 || die "the 'senbonzakura' command is not on PATH. Activate the
        environment the wheel is installed into first."

[ -d "$TRACK" ] || die "no track directory at '$TRACK'. It must hold bad_ds, good_ds and
        bad_eval_ds; build one with 'senbonzakura track'."
for part in bad_ds good_ds bad_eval_ds; do
  [ -e "$TRACK/$part" ] || die "the track at '$TRACK' has no '$part'. An abliteration run needs
        all three parts and would fail after loading the model."
done

mkdir -p "$OUT" || die "cannot create '$OUT'."
[ -w "$OUT" ] || die "'$OUT' is not writable."

RUN_LOG="$OUT/run.log"
say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$RUN_LOG" >&2; }

# `doctor` is RECORDED, never a gate here, and that distinction cost a release morning.
#
# It answers "can this install do everything senbonzakura claims", which is a superset of what
# this experiment needs. On the ROG it fails nine of fourteen checks: no llama-quantize, no
# vendored converter, no bundled track, no bundled corpora. Not one of those is used by an
# abliterate-then-grade run, which takes its corpus from --track and its benchmark from --eval,
# and blocking on them stopped two arms from starting on a machine that could have run them
# perfectly well. The gate was stricter than the job.
#
# What this run actually needs is checked above and below: the console script, the three track
# parts, a writable output, and room on the disk. Those are the preflight. This is a note.
doctor_rc=0
"$SZ" doctor >"$OUT/doctor.log" 2>&1 || doctor_rc=$?
if [ "$doctor_rc" -eq 0 ]; then
  say "NOTE  doctor is clean."
else
  say "NOTE  doctor reports problems (rc=$doctor_rc), recorded in $OUT/doctor.log and NOT
        blocking: it checks conversion, quantisation and the bundled corpora, none of which this
        experiment uses. If an arm fails later, read that log first."
fi

# Weights are kept by default, so the check is against holding every arm at once. A dry run is
# allowed to report the shortfall rather than refuse, since it writes nothing.
free_kb=$(df -Pk "$OUT" | awk 'NR==2 {print $4}')
: "${free_kb:=0}"
if [ "$free_kb" -lt 41943040 ] && [ "$DROP_WEIGHTS" -eq 0 ]; then
  msg="only $((free_kb / 1024 / 1024)) GiB free at '$OUT', and every arm's weights are kept by
        default. Five arms of even a small model run to tens of gigabytes. Free space, point
        --out at a bigger disk, or pass --drop-weights to keep only the artefacts (the weights
        are reproducible from the command each arm records; the artefacts are not)."
  if [ "$DRY_RUN" -eq 1 ]; then
    say "WOULD FAIL: $msg"
  else
    die "$msg"
  fi
fi

# ---------------------------------------------------------------- the arms
# One row per arm: the flags it adds to `abliterate`. 'stock' is scored, never edited.

arm_flags() {
  case "$1" in
    stock)            printf '' ;;
    single-pass)      printf -- '--method single-pass' ;;
    single-pass-raw)  printf -- '--method single-pass-raw' ;;
    searched)         printf -- '--method searched --trials %s' "$TRIALS" ;;
    searched-one-direction)
                      printf -- '--method searched-one-direction --trials %s' "$TRIALS" ;;
    searched-rounds4) printf -- '--method searched --trials %s --ablation-rounds 4' "$TRIALS" ;;
    searched-capgate) printf -- '--method searched --trials %s --capability-eval %s --capability-n %s --capability-task %s' \
                        "$TRIALS" "$EVAL_SET" "$PROBE_N" "$TASK" ;;
    *) return 1 ;;
  esac
}

for arm in ${ARMS//,/ }; do
  arm_flags "$arm" >/dev/null || die "unknown arm '$arm'. Known arms: ${KNOWN_ARMS//,/, }."
done

STOCK_CAP="$OUT/stock/capability.json"

# ---------------------------------------------------------------- the budget probe
# WHY THIS RUNS BEFORE ANYTHING ELSE
#
# A run on 2026-09-07 graded three arms at 41 of 200 answers ungradeable, 20.5%, on every arm
# INCLUDING the unedited reference, because --max-new 320 is not enough for a fifth of GSM8K. The
# answers that fail to finish are the LONG ones, so what got graded was an easier exam than the one
# set, and a 2.5 point drop was about to be read as a capability cost.
#
# `capability` now exits non-zero past its threshold, which stops the wrong number being collected
# but still costs a night. So the budget is SIZED first, on the stock arm alone, by doubling until
# it clears. One arm's generations to save six arms of unusable ones.
probe_budget() {
  local d="$OUT/_budget-probe" want="$MAX_NEW"
  mkdir -p "$d"
  for _try in 1 2 3 4; do
    say "budget probe: trying --max-new $want on the stock model"
    if timeout --signal=INT --kill-after=60 "$ARM_TIMEOUT"         "$SZ" capability --model "$MODEL" --device "$DEVICE" --eval "$EVAL_SET" --task "$TASK"           --n 40 --skip "$PROBE_N" --max-new "$want" --batch "$BATCH"           --label "budget-probe-$want" --out "$d/probe-$want.json" >>"$RUN_LOG" 2>&1; then
      say "budget probe: $want tokens clears the ungradeable threshold; using it for every arm"
      MAX_NEW="$want"
      return 0
    fi
    say "budget probe: $want tokens leaves too many answers unfinished"
    want=$(( want * 2 ))
  done
  say "budget probe: even $(( want / 2 )) tokens did not clear it. Using it anyway and every arm
        will report BUDGET, NOT MODEL, which is the honest outcome: this benchmark needs more
        room than this run is willing to spend, and that is a fact about the pairing rather than
        about any arm."
  MAX_NEW=$(( want / 2 ))
  return 0
}

if [ "$BUDGET_PROBE" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
  probe_budget
fi

run_step() {
  # run_step <marker> <description> <command...>
  local marker="$1" what="$2"; shift 2
  if [ -f "$marker" ] && [ "$FORCE" -eq 0 ]; then
    say "SKIP  $what (already done; --force to redo)"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    say "PLAN  $what"
    printf '        %s\n' "$*" >&2
    return 0
  fi
  say "START $what"
  local began; began=$(date +%s)
  # The exit code is captured from the command DIRECTLY, never read back through `$?` after an
  # `if`. `if cmd; then ...; fi` with no else evaluates to 0 when cmd FAILS, so the old form
  # reported every failure as "exit 0" and a run that ground to a halt read like a fast success.
  # Seventh time this project has measured an exit code through something that was not the
  # command, and the first where the mistake was in a file shipped the same day.
  #
  # The timeout is per step and it is not optional: a wedged generate loop would otherwise hold
  # the card until somebody noticed.
  local rc=0
  timeout --signal=INT --kill-after=120 "$ARM_TIMEOUT" "$@" >>"$RUN_LOG" 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    printf '%s\n' "$(date -u +%FT%TZ)" >"$marker"
    say "DONE  $what in $(( $(date +%s) - began ))s"
    return 0
  fi
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    say "TIMEOUT $what after ${ARM_TIMEOUT}s. Nothing marked done; re-run to resume."
  else
    say "FAILED  $what (exit $rc). See $RUN_LOG. Nothing marked done; re-run to resume."
  fi
  return "$rc"
}

failed_arms=""

for arm in ${ARMS//,/ }; do
  d="$OUT/$arm"
  mkdir -p "$d"

  if [ "$arm" != "stock" ]; then
    # shellcheck disable=SC2046  # word splitting of the flag list is the intent
    run_step "$d/.abliterated" "$arm: abliterate" \
      "$SZ" abliterate --model "$MODEL" --track "$TRACK" --device "$DEVICE" \
        --out "$d/model" $(arm_flags "$arm") || { failed_arms="$failed_arms $arm"; continue; }
    scored_model="$d/model"
  else
    scored_model="$MODEL"
  fi

  # --skip is the held-out boundary and it applies to EVERY arm, including the ones whose search
  # never saw the benchmark. The capgate arm's in-search probe reads the head of the eval set
  # (`questions[:n]`), so scoring it from the head would score it on the items it was selected on.
  # This project has already withdrawn three numbers that described the rows they were chosen on.
  # Skipping uniformly keeps every arm on the same held-out questions, which is what makes them
  # comparable at all.
  cap_cmd=("$SZ" capability --model "$scored_model" --device "$DEVICE"
           --eval "$EVAL_SET" --task "$TASK" --n "$N" --skip "$PROBE_N"
           --max-new "$MAX_NEW" --batch "$BATCH"
           --label "$arm" --out "$d/capability.json"
           --save-generations "$d/generations.jsonl")
  # The paired comparison is the number worth having, and it only exists once stock has run.
  if [ "$arm" != "stock" ] && [ -f "$STOCK_CAP" ]; then
    cap_cmd+=(--compare-to "$STOCK_CAP")
  elif [ "$arm" != "stock" ] && [ "$DRY_RUN" -eq 0 ]; then
    say "NOTE  $arm will be scored WITHOUT a paired comparison: $STOCK_CAP does not exist."
  fi

  run_step "$d/.scored" "$arm: capability" "${cap_cmd[@]}" \
    || { failed_arms="$failed_arms $arm"; continue; }

  if [ "$arm" != "stock" ] && [ "$DRY_RUN" -eq 0 ]; then
    run_step "$d/.card" "$arm: report card" \
      "$SZ" report --abliteration "$d/model/abliteration.json" \
        --capability "$d/capability.json" --out "$d/card.md" || true
    # Only ever on an explicit --drop-weights, and only once the arm is fully scored: deleting
    # the weights of an arm whose capability run failed would destroy the thing a re-run needs.
    if [ "$DROP_WEIGHTS" -eq 1 ] && [ -f "$d/.scored" ]; then
      find "$d/model" -name '*.safetensors' -delete 2>/dev/null || true
      say "NOTE  $arm: weights deleted at your request. Artefacts and generations kept."
    fi
  fi
done

if [ "$DRY_RUN" -eq 1 ]; then
  say "dry run only; nothing was executed."
  exit 0
fi

say "---- summary ----"
for arm in ${ARMS//,/ }; do
  f="$OUT/$arm/capability.json"
  if [ -f "$f" ]; then
    python3 - "$f" <<'PY' | tee -a "$RUN_LOG" >&2
import json, sys
d = json.load(open(sys.argv[1]))
s = d.get("summary", {})
line = f"  {d.get('label','?'):<18} {s.get('correct',0)}/{s.get('graded',0)} graded"
if s.get("indeterminate"):
    line += f", {s['indeterminate']} ungradeable"
c = d.get("change")
if c:
    lo, hi = c["delta_ci"]
    if c["distinguishable_from_zero"]:
        line += f"   change {c['delta_accuracy']:+.1%} [{lo:+.1%}, {hi:+.1%}]"
    else:
        line += f"   change not distinguishable from zero ({c['delta_accuracy']:+.1%})"
print(line)
PY
  else
    printf '  %-18s no result\n' "$arm" | tee -a "$RUN_LOG" >&2
  fi
done

if [ -n "$failed_arms" ]; then
  say "arms that did not finish:$failed_arms. Re-run the same command to resume them."
  exit 1
fi
say "all requested arms finished. Artefacts under $OUT."
