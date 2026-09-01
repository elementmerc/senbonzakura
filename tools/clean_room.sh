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
python -m build --wheel --outdir "$ROOT/dist-cleanroom" >/dev/null 2>&1 \
  || die "the wheel would not build, so there is nothing to install"
WHEEL="$(ls "$ROOT"/dist-cleanroom/*.whl | head -1)"
[ -n "$WHEEL" ] || die "no wheel was produced"
echo "clean room: $(basename "$WHEEL") ($(du -h "$WHEEL" | cut -f1))"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp "$WHEEL" "$ROOT/tools/clean_room_checks.py" "$WORK/"

# The install needs a network and the checks must not have one, so they are two containers and
# the installed tree is carried between them as a volume. A single container with the network
# left on would prove nothing about whether the corpora are really bundled.
INSTALL_FLAGS="--no-deps"
EXPECT="" ; [ "$MODE" = full ] && { INSTALL_FLAGS=""; EXPECT="--expect-torch"; }

script=$(cat <<EOF
set -e
python -m venv /home/box/venv
/home/box/venv/bin/pip install --quiet --upgrade pip
echo "installing (${MODE})..."
/home/box/venv/bin/pip install --quiet ${INSTALL_FLAGS} /box/$(basename "$WHEEL")
EOF
)

if [ -n "$HOST" ]; then
  echo "clean room: shipping to $HOST"
  ssh "$HOST" 'rm -rf ~/.senbon-cleanroom && mkdir -p ~/.senbon-cleanroom'
  scp -q "$WORK"/* "$HOST:~/.senbon-cleanroom/"
  # Install (network on, nothing else), then check (network off). Two invocations, one volume.
  ssh "$HOST" "docker run --rm --user \$(id -u):\$(id -g) -v ~/.senbon-cleanroom:/box -e HOME=/home/box \
       --tmpfs /home/box:rw,exec,size=4g -w /box $IMAGE sh -c '$script && cp -r /home/box/venv /box/venv'" \
    || die "the install failed on $HOST"
  ssh "$HOST" "docker run --rm --user "$(id -u):$(id -g)" --network none --cap-drop ALL --security-opt no-new-privileges \
       -v ~/.senbon-cleanroom:/box:ro -e HOME=/home/box --tmpfs /home/box:rw,size=64m -w /tmp \
       $IMAGE /box/venv/bin/python /box/clean_room_checks.py $EXPECT"
  rc=$?
  ssh "$HOST" 'rm -rf ~/.senbon-cleanroom'
  exit $rc
fi

docker run --rm --user "$(id -u):$(id -g)" -v "$WORK:/box" -e HOME=/home/box --tmpfs /home/box:rw,exec,size=4g \
  -w /box "$IMAGE" sh -c "$script && cp -r /home/box/venv /box/venv" || die "the install failed"
docker run --rm --user "$(id -u):$(id -g)" --network none --cap-drop ALL --security-opt no-new-privileges \
  -v "$WORK:/box:ro" -e HOME=/home/box --tmpfs /home/box:rw,size=64m -w /tmp \
  "$IMAGE" /box/venv/bin/python /box/clean_room_checks.py $EXPECT
