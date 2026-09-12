#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# clean_room.sh — install the wheel on a machine that has never seen this project, and find out
# what it can actually do.
#
# WHY THIS EXISTS
#
# Every check that runs in the checkout runs on a machine that already has the vendored binaries,
# the corpora sources, the git history and the dev dependencies sitting on disk. A stranger has
# none of those. The wheel ships no llama.cpp binaries, so `pip install senbonzakura` cannot
# convert or quantise, and nothing said so until somebody ran `doctor` by hand on a second box.
#
# THE CONTAINER IS THE CONTROL, AND THESE FLAGS ARE THE POINT
#
#   --network none      after the install. Proves the bundled corpora really are bundled, and that
#                       nothing is fetched at first use. The install step is the ONLY step allowed
#                       a network, and it is a separate container.
#   --rm                nothing is left behind. This machine runs the operator's real services and
#                       a test that accumulates images would eventually be a test that fills a disk.
#   -e nothing          no HF token, no ssh agent, no AWS keys. The checks assert their absence
#                       rather than trusting this line.
#   --read-only         with an explicit tmpfs, so the image cannot be mutated into something
#                       other than what was built.
#   no capabilities     --cap-drop ALL, --security-opt no-new-privileges.
#
# USAGE
#
#   tools/clean_room.sh                 # fast: --no-deps, proves the CLI works without torch
#   tools/clean_room.sh --full          # the real clean-machine install, torch and all. Slow.
#   tools/clean_room.sh --host atlas    # run it on another machine over ssh
#
set -euo pipefail

MODE=fast
HOST=""
IMAGE="python:3.13-slim"
while [ $# -gt 0 ]; do
  case "$1" in
    --full)  MODE=full; shift ;;
    --host)  HOST="${2:?--host needs a name}"; shift 2 ;;
    --image) IMAGE="${2:?--image needs a name}"; shift 2 ;;
    -h|--help) sed -n '3,32p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

die() { echo "clean room: $*" >&2; exit 1; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

command -v docker >/dev/null 2>&1 || [ -n "$HOST" ] || die "docker is not installed here. Use --host."

echo "clean room: building the wheel"
rm -rf "$ROOT/dist-cleanroom"
python -m build --wheel --outdir "$ROOT/dist-cleanroom" checker >/dev/null 2>&1 || {
  echo "clean room: could not build the checker wheel" >&2; exit 1; }
python -m build --wheel --outdir "$ROOT/dist-cleanroom" >/dev/null 2>&1 \
  || die "the wheel would not build, so there is nothing to install"
# TWO WHEELS SINCE Q-29. `senbonzakura` depends on `senbonzakura-check`, which is not on PyPI
# yet, so the full install cannot resolve without the local one beside it. The senbonzakura
# wheel is the one under test; the checker wheel is carried in so the dependency resolves.
CHECK_WHEEL="$(ls "$ROOT"/dist-cleanroom/senbonzakura_check-*.whl | head -1)"
WHEEL="$(ls "$ROOT"/dist-cleanroom/*.whl | grep -v senbonzakura_check | head -1)"
[ -n "$WHEEL" ] || die "no wheel was produced"
echo "clean room: $(basename "$WHEEL") ($(du -h "$WHEEL" | cut -f1))"

# WHERE THE WORK DIRECTORY LIVES, and it is a pre-flight rather than a hope.
#
# A full install is torch and its dependencies, several gigabytes, and `mktemp -d` lands in
# /tmp, which on this laptop is a 3.5 GB tmpfs: RAM. The run then died with "Disk quota
# exceeded" partway through, having proved nothing about the wheel, and the message looked like
# the install failing rather than the harness running out of room. The 2026-09-01 fix moved the
# venv INSIDE the container off its tmpfs and did not touch the host directory that container
# mounts, so it was the same defect one layer out.
#
# So the space is measured before anything is built, and a directory that cannot hold the
# install is refused with the numbers rather than discovered halfway through.
NEED_MB=1024
[ "$MODE" = full ] && NEED_MB=12288

free_mb() { df -Pm "$1" 2>/dev/null | awk 'NR==2 {print $4}'; }

WORK=""
for base in ${SENBON_CLEANROOM_TMPDIR:-} "${TMPDIR:-/tmp}" /var/tmp "$ROOT/.cleanroom-work"; do
  [ -n "$base" ] || continue
  mkdir -p "$base" 2>/dev/null || continue
  have="$(free_mb "$base")"
  [ -n "$have" ] || continue
  if [ "$have" -ge "$NEED_MB" ]; then
    WORK="$(mktemp -d "$base/senbon-cleanroom.XXXXXX")" || continue
    echo "clean room: working in $base (${have} MB free, needs ~${NEED_MB} MB for --${MODE})"
    break
  fi
  echo "clean room: skipping $base, ${have} MB free against ~${NEED_MB} MB needed" >&2
done
[ -n "$WORK" ] || die "no directory with ~${NEED_MB} MB free for a --${MODE} run. Point
  SENBON_CLEANROOM_TMPDIR at one with room. This is the harness running out of space, not the
  wheel failing: nothing about the wheel has been tested."
trap 'rm -rf "$WORK"' EXIT
cp "$WHEEL" "$CHECK_WHEEL" "$ROOT/tools/clean_room_checks.py" "$WORK/"

# The install needs a network and the checks must not have one, so they are two containers and
# the installed tree is carried between them as a volume. A single container with the network
# left on would prove nothing about whether the corpora are really bundled.
INSTALL_FLAGS="--no-deps"
EXPECT="" ; [ "$MODE" = full ] && { INSTALL_FLAGS=""; EXPECT="--expect-torch"; }

# The venv is built ON THE MOUNTED VOLUME, which is disk, not in a tmpfs. A full install is torch
# and its dependencies, several gigabytes, and a tmpfs is RAM: the first attempt died on
# "No space left on device" against a 4 GB tmpfs on a 28 GB machine, which is a harness limit
# being reported as if the install had failed.
script=$(cat <<EOF
set -e
rm -rf /box/venv
python -m venv /box/venv
/box/venv/bin/pip install --quiet --upgrade pip
echo "installing (${MODE})..."
/box/venv/bin/pip install --quiet --no-deps /box/$(basename "$CHECK_WHEEL")
/box/venv/bin/pip install --quiet ${INSTALL_FLAGS} /box/$(basename "$WHEEL")
EOF
)

if [ -n "$HOST" ]; then
  echo "clean room: shipping to $HOST"
  ssh "$HOST" 'rm -rf ~/.senbon-cleanroom && mkdir -p ~/.senbon-cleanroom'
  scp -q "$WORK"/* "$HOST:~/.senbon-cleanroom/"
  # Install (network on, nothing else), then check (network off). Two invocations, one volume.
  ssh "$HOST" "docker run --rm --user \$(id -u):\$(id -g) -v ~/.senbon-cleanroom:/box -e HOME=/home/box \
       --tmpfs /home/box:rw,size=64m -w /box $IMAGE sh -c '$script'" \
    || die "the install failed on $HOST"
  ssh "$HOST" "docker run --rm --user "$(id -u):$(id -g)" --network none --cap-drop ALL --security-opt no-new-privileges \
       -v ~/.senbon-cleanroom:/box:ro -e HOME=/home/box --tmpfs /home/box:rw,size=64m -w /tmp \
       $IMAGE /box/venv/bin/python /box/clean_room_checks.py $EXPECT"
  rc=$?
  ssh "$HOST" 'rm -rf ~/.senbon-cleanroom'
  exit $rc
fi

docker run --rm --user "$(id -u):$(id -g)" -v "$WORK:/box" -e HOME=/home/box --tmpfs /home/box:rw,size=64m \
  -w /box "$IMAGE" sh -c "$script" || die "the install failed"
docker run --rm --user "$(id -u):$(id -g)" --network none --cap-drop ALL --security-opt no-new-privileges \
  -v "$WORK:/box:ro" -e HOME=/home/box --tmpfs /home/box:rw,size=64m -w /tmp \
  "$IMAGE" /box/venv/bin/python /box/clean_room_checks.py $EXPECT
