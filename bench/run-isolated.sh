#!/usr/bin/env bash
#
# run-isolated.sh — run somebody else's abliteration tool without handing it anything.
#
# The Dockerfile is not the control. This file is. A container started with the wrong flags is a
# process with your credentials and a network, wearing a container for decoration, so every
# invariant below is asserted here rather than assumed:
#
#   1. NO NETWORK.               `--network none`. The strongest control by a distance: a tool that
#                                cannot reach the network cannot exfiltrate anything, whatever it
#                                finds. It also forces the run to be reproducible, because nothing
#                                can be fetched mid-run that was not fetched before it.
#   2. NO CREDENTIALS.           No HF token, no ssh agent, no ~/.cache mount, no environment
#                                passed through. Checked below rather than trusted, because the
#                                natural way to make a failing run work is to mount one more thing.
#   3. READ-ONLY INPUTS.         Weights and corpus are mounted `:ro`. A benchmark that lets a tool
#                                modify the weights the next tool will read is not a benchmark.
#   4. READ-ONLY ROOT.           `--read-only` with an explicit tmpfs, so the image cannot be
#                                mutated into something different from what was built.
#   5. NO NEW PRIVILEGES.        `--security-opt no-new-privileges`, capabilities dropped.
#   6. BOUNDED.                  Memory, pids and a wall-clock timeout, so a runaway cannot take the
#                                card or the box with it.
#
# Everything the tool needs must therefore be staged BEFORE the run, by us, deliberately. That is
# the intended friction: it makes the inputs explicit, which is the same property the benchmark
# needs to be reproducible by a stranger.
#
# Usage:
#   bench/run-isolated.sh --tool heretic --ref v1.2.3 --model /models/Qwen3-1.7B \
#                         --out /work/out/heretic-seed42 -- <command to run inside>
set -euo pipefail

IMAGE="${BENCH_IMAGE:-senbon-bench:tool}"
TIMEOUT="${BENCH_TIMEOUT:-21600}"          # 6h; a search that runs longer is a hang, not progress
MEM="${BENCH_MEM:-24g}"
PIDS="${BENCH_PIDS:-512}"

TOOL="" REF="" MODEL="" CORPUS="" OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --tool)   TOOL="$2"; shift 2;;
    --ref)    REF="$2"; shift 2;;
    --model)  MODEL="$2"; shift 2;;
    --corpus) CORPUS="$2"; shift 2;;
    --out)    OUT="$2"; shift 2;;
    --)       shift; break;;
    *) echo "run-isolated: unknown argument $1" >&2; exit 2;;
  esac
done

die() { echo "run-isolated: $*" >&2; exit 2; }

[ -n "$TOOL" ] || die "--tool is required; it names the row this run produces"
[ -n "$MODEL" ] || die "--model is required and must already exist on disk (nothing is downloaded)"
[ -n "$OUT" ]   || die "--out is required"
[ $# -gt 0 ]    || die "no command given after --"
[ -d "$MODEL" ] || die "model directory $MODEL does not exist. Weights are staged BEFORE the run, deliberately, because the container has no network to fetch them with."
[ -n "$CORPUS" ] && { [ -d "$CORPUS" ] || die "corpus directory $CORPUS does not exist"; }

mkdir -p "$OUT"

# Refuse to run if a credential is sitting in the environment we are about to be careful about.
# Not because it would be passed in (it would not, nothing is passed in), but because its presence
# means this shell is the credential-holding shell, and the next person to debug a failing run will
# reach for `-e HF_TOKEN` to make it work. Saying so here is cheaper than discovering it later.
for var in HF_TOKEN HUGGING_FACE_HUB_TOKEN RUNPOD_API_KEY AWS_SECRET_ACCESS_KEY GITHUB_TOKEN; do
  if [ -n "${!var:-}" ]; then
    echo "run-isolated: NOTE: $var is set in this shell. It is NOT passed to the container, and" >&2
    echo "  it must not be. If a run fails for want of a credential, stage the input instead." >&2
  fi
done

# WSL2 exposes the GPU as a device plus the driver libraries; no container toolkit involved.
GPU_ARGS=()
if [ -e /dev/dxg ] && [ -d /usr/lib/wsl/lib ]; then
  # BOTH mounts are required and the second is easy to miss. With only /usr/lib/wsl/lib the
  # container loads libcuda.so.1 successfully, reports "Can't initialize NVML", and returns
  # device_count 0 with is_available() False. That reads as "no GPU in this container" rather than
  # as "one bind mount short", and the driver store is what libcuda needs to reach /dev/dxg.
  GPU_ARGS=(--device=/dev/dxg
            -v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro
            -v /usr/lib/wsl/drivers:/usr/lib/wsl/drivers:ro)
elif [ -e /dev/nvidia0 ]; then
  GPU_ARGS=(--gpus all)                    # a normal Linux host with the toolkit installed
else
  die "no GPU found: neither /dev/dxg (WSL2) nor /dev/nvidia0. A CPU-only abliteration run would take days and would not be comparable to a GPU one, so this refuses rather than quietly producing an incomparable row."
fi

CORPUS_ARGS=()
[ -n "$CORPUS" ] && CORPUS_ARGS=(-v "$CORPUS:/corpus:ro")

# Provenance beside the result, not in a tag. A row whose tool version is unknown cannot be
# defended when its author disputes it, and this benchmark is published with an invitation to
# dispute it.
cat > "$OUT/run-meta.json" <<META
{
  "tool": "$TOOL",
  "tool_ref": "${REF:-unspecified}",
  "image": "$IMAGE",
  "image_digest": "$(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE" 2>/dev/null || docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || echo unknown)",
  "model": "$MODEL",
  "corpus": "${CORPUS:-none}",
  "network": "none",
  "command": "$*"
}
META

echo "run-isolated: $TOOL${REF:+ @ $REF} on $(basename "$MODEL"), no network, inputs read-only"

# Prove the GPU is reachable INSIDE the sealed box before spending hours in it. Without the driver
# store mount this returns False while every other flag looks right, and the arm would run on CPU,
# take a day, and produce a runtime column that is not comparable with anything.
if ! docker run --rm --network none "${GPU_ARGS[@]}" "$IMAGE" \
     python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  die "the GPU is not visible inside the container. Check that /usr/lib/wsl/drivers is mounted as well as /usr/lib/wsl/lib; with only the latter, libcuda loads but reports no devices."
fi

# `timeout --signal=TERM --kill-after` so a hung run is killed rather than left holding the card.
set +e
timeout --signal=TERM --kill-after=60 "$TIMEOUT" \
docker run --rm \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=2g \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  --memory "$MEM" \
  --pids-limit "$PIDS" \
  "${GPU_ARGS[@]}" \
  -v "$MODEL:/model:ro" \
  "${CORPUS_ARGS[@]}" \
  -v "$OUT:/work/out:rw" \
  --tmpfs /work/cache:rw,size=4g \
  -e TOOL_REF="${REF:-}" \
  "$IMAGE" "$@"
rc=$?
set -e

# 124 is timeout's own code. Distinguished because "the search was still running at six hours" and
# "the tool crashed" are different rows in a benchmark and must not be recorded as one.
if [ "$rc" -eq 124 ]; then
  echo "run-isolated: TIMED OUT after ${TIMEOUT}s" | tee -a "$OUT/run.log" >&2
  printf 'TIMEOUT\n' > "$OUT/STATUS"
elif [ "$rc" -ne 0 ]; then
  echo "run-isolated: exited $rc" | tee -a "$OUT/run.log" >&2
  printf 'FAILED %s\n' "$rc" > "$OUT/STATUS"
else
  printf 'OK\n' > "$OUT/STATUS"
fi
exit "$rc"
