#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# repair_manylinux.sh — turn the platform wheel into one PyPI will accept, and prove it runs.
#
# WHY THIS EXISTS (decision Q-34, 2026-09-12)
#
# `python -m build` with the binaries vendored produces `py3-none-linux_x86_64`. PyPI refuses
# that tag outright, and it refuses it at UPLOAD, which in this project's release flow happens
# after the GitHub Release is published and the tag is pushed. The version number is then burned.
#
# A bare `linux_x86_64` tag is also dishonest in a way that costs a user rather than us: it says
# nothing about which glibc the binaries inside need, so an installer cannot tell whether they
# will run. Measured on 2026-09-12: the wheel's own `llama-quantize` needs `libgomp.so.1`, which
# `python:3.13-slim` does not ship, so the clean room found a wheel that installed happily and
# carried a quantiser that could not start.
#
# `auditwheel repair` fixes both. It bundles the libraries the binaries need but the platform
# does not guarantee (measured: libssl, libcrypto, libgomp), rewrites the RPATH of every ELF
# INCLUDING THE EXECUTABLES so they find them, and retags the wheel to the floor it actually
# satisfies.
#
# WHAT WAS MEASURED, so the next person does not have to rediscover it
#
#   * The floor is NOT set by glibc. The binaries top out at GLIBC_2.34, but they need
#     GLIBCXX_3.4.30, which is GCC 12, and that is what forces manylinux_2_35.
#   * Runs after repair: python:3.13-slim, ubuntu 22.04, ubuntu 24.04, debian 12, fedora 40,
#     archlinux. Fails on almalinux 9 (glibc 2.34, GLIBCXX_3.4.29), which is the RHEL 9 family
#     and is BELOW the tag, so pip there refuses the wheel rather than installing a broken one.
#     Verified: "not a supported wheel on this platform".
#   * The repair is what fixes libgomp. On an image with no libgomp at all, `doctor` on the
#     repaired wheel reports "llama-quantize  vendored, runs".
#
# THE TAG IS DERIVED, NOT ASSERTED. `--plat auto` asks auditwheel what the wheel satisfies, so an
# upstream llama.cpp build raising its floor changes the answer instead of producing a wheel that
# claims a compatibility it lost. EXPECTED_FLOOR below only makes that change LOUD.
#
# USAGE
#
#   tools/repair_manylinux.sh dist/senbonzakura-<v>-py3-none-linux_x86_64.whl
#   tools/repair_manylinux.sh <wheel> --skip-runtime-check     # no second container
#
set -euo pipefail

# Pinned by digest, never by tag (baseline Section 5). The image supplies auditwheel and patchelf
# and nothing from it reaches the wheel except the libraries auditwheel vendors, which is why the
# digest matters: those libraries ship to users.
IMAGE="quay.io/pypa/manylinux_2_34_x86_64@sha256:934419fa742ff922855f464203cc0ad5afe3317a8541712a00cf3dc7fb893fc8"

#: What the repair produced when this was written. Not a requirement: a MISMATCH is reported so a
#: silent move in the floor is visible, because the floor decides which users can install at all.
EXPECTED_FLOOR="manylinux_2_35_x86_64"

#: An image with no libgomp, which is where the undeclared dependency was found.
RUNTIME_IMAGE="python:3.13-slim"

die() { echo "repair: $*" >&2; exit 1; }

WHEEL=""
RUNTIME_CHECK=1
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-runtime-check) RUNTIME_CHECK=0; shift ;;
    -h|--help) sed -n '3,45p' "$0"; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) WHEEL="$1"; shift ;;
  esac
done

[ -n "$WHEEL" ] || die "name the wheel to repair. See --help."
[ -f "$WHEEL" ] || die "$WHEEL does not exist"
command -v docker >/dev/null 2>&1 || die "docker is not installed, and the repair must run inside
  the pinned manylinux image. Running auditwheel on THIS machine would vendor this machine's
  libraries into a wheel other people install, and would read this machine's glibc as the floor:
  measured 2026-09-12, the same wheel reads manylinux_2_38 on a modern host and manylinux_2_35
  inside the image."

case "$(basename "$WHEEL")" in
  *-linux_x86_64.whl) ;;
  *manylinux*) die "$(basename "$WHEEL") is already repaired." ;;
  *) die "$(basename "$WHEEL") is not a bare linux_x86_64 wheel, so there is nothing to repair.
  A universal wheel does not carry binaries and does not need this." ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IN_DIR="$(cd "$(dirname "$WHEEL")" && pwd)"
OUT_DIR="$ROOT/dist-manylinux"
rm -rf "$OUT_DIR"; mkdir -p "$OUT_DIR"

echo "repair: $(basename "$WHEEL")"
echo "repair: inside $IMAGE"

docker run --rm \
  -v "$IN_DIR":/in:ro -v "$OUT_DIR":/out \
  "$IMAGE" bash -lc "
    set -e
    /opt/python/cp312-cp312/bin/pip install --quiet auditwheel
    /opt/python/cp312-cp312/bin/auditwheel repair --plat auto -w /out '/in/$(basename "$WHEEL")'
  " 2>&1 | grep -vE "^(WARNING: Running pip|INFO:auditwheel.wheel_abi:setting)" || true

REPAIRED="$(ls "$OUT_DIR"/*.whl 2>/dev/null | head -1)" || true
[ -n "$REPAIRED" ] || die "auditwheel produced no wheel. Read the output above; the usual cause is
  a shared library in the purelib folder, which means the distribution was built pure. See
  setup.py's _BinaryDistribution."

TAG="$(basename "$REPAIRED" | sed -E 's/.*-py3-none-(.*)\.whl/\1/')"
echo "repair: tag $TAG"
if [ "$TAG" != "$EXPECTED_FLOOR" ]; then
  echo "repair: NOTE, the floor moved. It was $EXPECTED_FLOOR when this script was written and is
  now $TAG. That is not an error and the wheel is fine, but it changes WHO CAN INSTALL: a higher
  floor drops distributions, a lower one adds them. Update EXPECTED_FLOOR and say in the commit
  which way it went and why." >&2
fi

# The repair is only useful if it reached the EXECUTABLES. auditwheel's headline job is shared
# objects, and a repair that bundled the libraries while leaving `llama-quantize` looking for the
# system copy would pass every tag check and fail at first use, which is the exact defect the
# unrepaired wheel had.
python3 - "$REPAIRED" <<'PY'
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
vendored = [n for n in names if ".libs/" in n and n.endswith((".so", ".so.1")) or ".libs/lib" in n]
binaries = [n for n in names if "/vendor/bin/" in n]
if not vendored:
    sys.exit("repair: the wheel carries no vendored libraries, so nothing was bundled. If the "
             "binaries genuinely need nothing beyond the manylinux whitelist this is fine, but "
             "it was not true when this was written (libssl, libcrypto and libgomp were all "
             "bundled), so check before believing it.")
if not binaries:
    sys.exit("repair: the repaired wheel carries no vendored binaries at all.")
print(f"repair: {len(vendored)} bundled library file(s), {len(binaries)} vendored binary file(s)")
PY

if [ "$RUNTIME_CHECK" = 1 ]; then
  # THE CHECK THAT MATTERS. Not "does it have the right tag" but "does the binary start on a
  # machine that lacks what it needs", which is the question the tag exists to answer.
  echo "repair: starting the quantiser on $RUNTIME_IMAGE, which has no libgomp"
  WORK="$(mktemp -d "${TMPDIR:-/var/tmp}/senbon-repair.XXXXXX")"
  trap 'rm -rf "$WORK"' EXIT
  ( cd "$WORK" && unzip -q "$REPAIRED" )
  # CAPTURED, NOT PIPED. `docker ... | head -1 | grep -q` looks equivalent and is not: `head`
  # closes the pipe, docker dies of SIGPIPE with status 141, and `set -o pipefail` propagates
  # that even though grep matched. The first version of this check reported a perfectly good
  # wheel as unpublishable for that reason, which is the same exit-code-through-a-pipe trap
  # this project has now hit four times.
  started="$(docker run --rm -v "$WORK":/w:ro "$RUNTIME_IMAGE" \
       /w/senbonzakura/vendor/bin/linux-x86_64/llama-quantize --help 2>&1 || true)"
  if printf '%s\n' "$started" | head -1 | grep -q "^usage:"; then
    echo "repair: it runs where the unrepaired wheel could not"
  else
    echo "$started" | head -3 >&2
    die "the repaired quantiser does not start on $RUNTIME_IMAGE. The tag says the wheel is
  installable there, so this wheel would install and then fail at first use, which is worse than
  refusing to install. Do not publish it."
  fi
fi

echo "repair: $REPAIRED"
