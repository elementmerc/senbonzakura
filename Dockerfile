# SPDX-License-Identifier: AGPL-3.0-or-later
#
# senbonzakura, as a release artefact you can run without installing anything.
#
# WHAT THIS IMAGE IS FOR, AND WHY IT IS NOT THE WHEEL
#
# `pip install senbonzakura` gives you a pure-Python wheel that CANNOT convert or quantise,
# because the llama.cpp binaries are not in it and cannot portably be: they are per-platform
# executables and the wheel is `py3-none-any`. That is a real gap and users hit it as a `doctor`
# report saying the install cannot do what it claims.
#
# This image closes it. The binaries are vendored at build time from the committed pin manifest,
# hash-verified by the same tool the developers use, and smoke-tested before the image is
# finished. So `convert`, `quantise`, `imatrix` and `doctor` work the moment it starts.
#
# CPU BY DEFAULT, AND THAT IS A DELIBERATE LIMIT
#
# torch arrives from the CPU index. The image is therefore 2.11 GB (measured, not estimated)
# rather than the 6 GB or so a CUDA build costs, and it
# does everything that is CPU-bound: convert, quantise, importance matrices, scoring, and the
# abliteration of small models slowly. It CANNOT use a GPU even with `--gpus all`, because a CPU
# torch build has no CUDA kernels, and `doctor` will tell you so rather than pretending.
#
# A CUDA variant is a separate image and a separate decision, not a flag on this one.
#
# BUILD
#   docker build -t senbonzakura:0.3.0 .
#
# RUN  (the model and the output live on your machine, not in the container)
#   docker run --rm -v "$PWD:/work" senbonzakura:0.3.0 doctor
#   docker run --rm -v "$PWD:/work" senbonzakura:0.3.0 quantise model-f16.gguf --type Q4_K_M
#
# The build needs a network, to fetch dependencies and the pinned binaries. Running does not:
# the corpora are bundled and nothing is fetched at first use.
#
# "The corpora are bundled" IS A CONDITION ON THE BUILD, not a property of this file, and until
# 2026-09-23 it was simply false in CI. They are generated rather than committed, so they have to
# exist in the build context before `docker build` runs:
#
#     python tools/packaging/build_corpora.py   # needs an authenticated gh
#     python tools/packaging/pack_track.py
#
# The runtime stage now refuses the build when they are missing, rather than producing an image
# that starts, answers `--help` and fails `--track default` for everybody who pulls it.

# ── builder ─────────────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

# `curl` for the pinned binary download, `git` because the vendoring tool records provenance.
# libgomp1 is not optional: llama.cpp's binaries are linked against OpenMP, and `slim` does not
# carry it. Without it llama-quantize exits 127 on a missing libgomp.so.1, which the vendoring
# tool's smoke check catches at build time rather than letting the image ship a binary that
# cannot start.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md LICENSE THIRD-PARTY-NOTICES.md THIRD-PARTY-CORPORA.md \
     APACHE-2.0.txt ACCEPTABLE-USE.md ./
COPY src/ ./src/
# The checker's own tree, because `senbonzakura` depends on `senbonzakura-check` and that name is
# not on PyPI yet. Without this the pip line below reports "Invalid requirement: './checker'",
# which is pip saying the path does not exist rather than anything being wrong with the package.
COPY checker/ ./checker/
COPY tools/ ./tools/
COPY man/ ./man/

# INSTALLED BEFORE THE VENDORING, and the order is load-bearing: the vendored
# `convert_hf_to_gguf.py` opens with `import torch`, and the vendoring tool smoke-tests it before
# recording it. Vendoring first fails with a bare ModuleNotFoundError, which is the tool refusing
# to record a converter it could not prove runs.
#
# CPU torch explicitly. Without the index the default wheel pulls the CUDA runtime and several
# gigabytes of libraries this image has told the user it will not use.
# The checker is installed FIRST, from this tree. `senbonzakura` depends on
# `senbonzakura-check` by name (decision Q-29) and that name is not on PyPI yet, so a plain
# install cannot resolve until it is. `--no-deps` on it is honest rather than defensive: the
# distribution declares no dependencies at all, which is the property it exists to hold.
RUN python -m pip install --no-cache-dir --quiet \
      --index-url https://download.pytorch.org/whl/cpu torch \
 && python -m pip install --no-cache-dir --quiet --no-deps ./checker \
 && python -m pip install --no-cache-dir --quiet ".[abliterate]"

# The pinned llama.cpp binaries, AFTER torch exists so the converter's smoke test can run.
# Fetched and hash-checked by the same tool the developers use: it refuses on a hash mismatch, so
# a build cannot quietly ship a binary nobody verified. The vendored tree is not carried by
# `pip install` (package-data excludes `vendor/bin/**` on purpose, because a `py3-none-any` wheel
# has no business holding a platform executable), so the runtime stage copies it explicitly.
RUN python -m pip install --no-cache-dir --quiet "requests>=2" \
 && python tools/packaging/vendor_llama.py \
 && test -x src/senbonzakura/vendor/bin/linux-x86_64/llama-quantize \
 && test -x src/senbonzakura/vendor/bin/linux-x86_64/llama-imatrix

# ── runtime ─────────────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="senbonzakura" \
      org.opencontainers.image.description="Precision abliteration, with receipts." \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/elementmerc/senbonzakura" \
      org.opencontainers.image.documentation="https://elementmerc.github.io/senbonzakura"

# libgomp1 is the OpenMP runtime llama.cpp's binaries link against, needed in the stage that
# actually runs them. The user exists because an abliteration writes gigabytes into a mounted
# directory, and a tool that writes them as uid 0 leaves a person unable to delete their own
# output.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 senbon \
 && mkdir -p /work \
 && chown senbon:senbon /work
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin/senbonzakura /usr/local/bin/senbonzakura
# The binaries are copied EXPLICITLY, because `pip install` will not carry them: package-data
# declares `data/*.bin` and `vendor/*.json` and deliberately not `vendor/bin/**`, since a
# `py3-none-any` wheel has no business containing a platform executable. That is exactly the gap
# this image exists to close, so it is closed here rather than by loosening the wheel.
COPY --from=builder /build/src/senbonzakura/vendor/bin \
     /usr/local/lib/python3.13/site-packages/senbonzakura/vendor/bin
COPY --from=builder /build/src/senbonzakura/vendor/src \
     /usr/local/lib/python3.13/site-packages/senbonzakura/vendor/src

USER senbon
WORKDIR /work

# Hugging Face writes here; without it the cache lands somewhere unwritable and the failure is a
# permissions error a long way from its cause.
ENV HF_HOME=/work/.cache/huggingface \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Proven at build time rather than asserted in a comment: if the binaries did not survive the copy
# into the runtime stage, or cannot start for want of a shared library, the BUILD fails rather
# than the user finding out. `|| true` would make this decoration, so there is none: the grep is
# the assertion, and `doctor`'s own exit code is not used because a CPU-only image legitimately
# carries a torch advisory and would exit non-zero for a reason that is not a fault.
RUN senbonzakura doctor 2>&1 | tee /tmp/doctor.txt; \
    grep -q "llama-quantize *vendored, runs" /tmp/doctor.txt \
      || { echo "the image ships a quantiser that does not run:"; cat /tmp/doctor.txt; exit 1; }
# AND NOTHING ELSE IS FAILING, which the grep above could not see and therefore did not.
#
# On 2026-09-23 the first real run of the distribute workflow published an image whose `doctor`
# reported 7 of 19 checks failed: the bundled track and all six corpora were absent. The build
# above passed, because it asserts one line about the quantiser and reads nothing else, and the
# image job asserted the same one line. Two gates, one blind spot, shared.
#
# The corpora and the track are GENERATED rather than committed, so a plain `docker build` from a
# clean clone has nothing to copy and produces exactly that image. The right answer is to refuse
# the build and say what to run, rather than to ship something that starts, answers `--help`, and
# then fails `--track default` for everybody. `.dockerignore` already says an image is a release
# artefact like a wheel and should carry them; this is the check that it does.
#
# doctor's own exit code is still not used on its own: a CPU-only image legitimately carries a
# torch advisory and would exit non-zero for a reason that is not a fault. The count is the
# assertion.
#
# THE BUNDLED TRACK IS THE ONE NAMED EXCEPTION, and it is named rather than tolerated by a
# loosened count. It is packed from a HELD-OUT track the operator keeps off this repository
# (RELEASING.md, "Before anything else: pack the evaluation track"), so no CI job can build one
# and a build that demanded it would fail every push forever. Its absence means `--track default`
# does not work in an image built this way, which is a real limitation of the dev lane rather
# than a check to switch off; it is recorded in DEFERRED.md as an open question.
# A DELIBERATELY CODE-ONLY BUILD IS A REAL THING AND IS NOT A RELEASE.
#
# The check below and the CI job that runs it disagreed, and the job was the one that broke.
# `.github/workflows/ci.yml` builds this image on a runner ON PURPOSE without corpora, to prove
# the property a runner CAN prove: that an image lacking them says so rather than quietly
# returning rows from somewhere. Its own comment says as much at length. Then this gate arrived
# and refused to build that image at all, so the job asserting "the absence is legible" could no
# longer get an image whose absence it could read.
#
# So the refusal is opt-out, by a build argument, rather than absolute. The default is still to
# refuse, which is what protects a release: nothing changes for anyone who types `docker build`.
# CI passes `--build-arg ALLOW_CODE_ONLY=1` and says why at the call site. An argument is used
# rather than loosening the count because a code-only build is a DECLARED intent, and a reader of
# either file can see which kind of image they are looking at.
ARG ALLOW_CODE_ONLY=0
RUN grep -vE "^\s+✗\s+bundled track" /tmp/doctor.txt > /tmp/doctor-checked.txt; \
    if [ "${ALLOW_CODE_ONLY}" = "1" ]; then \
      grep -vE "^\s+✗\s+corpus " /tmp/doctor-checked.txt > /tmp/doctor-kept.txt; \
      mv /tmp/doctor-kept.txt /tmp/doctor-checked.txt; \
      echo "ALLOW_CODE_ONLY=1: this image is built without corpora on purpose."; \
      echo "It cannot serve --track default and is not a release artefact."; \
    fi; \
    if grep -qE "^\s+✗" /tmp/doctor-checked.txt; then \
      echo "----------------------------------------------------------------"; \
      echo "This image cannot do what it claims. doctor reports:"; \
      grep -E "^\s+✗" /tmp/doctor.txt || true; \
      echo ""; \
      echo "The bundled data is generated, not committed, so a build from a clean"; \
      echo "clone has nothing to copy. Build it first, in the source tree:"; \
      echo "    python tools/packaging/build_corpora.py   # needs an authenticated gh"; \
      echo "then build the image again."; \
      echo ""; \
      echo "The bundled TRACK is separate and is not checked here: it is packed"; \
      echo "from a held-out track that lives outside this repository, with"; \
      echo "    python tools/packaging/pack_track.py --track <your held-out track>"; \
      echo "----------------------------------------------------------------"; \
      exit 1; \
    fi
RUN senbonzakura head-to-head --help >/dev/null \
 && senbonzakura quantise --help >/dev/null \
 && echo "the delegated commands start"
# /work is the default working directory and HF_HOME lives under it. It was created by WORKDIR as
# root, so the unprivileged user could not write to it and every download failed on a permissions
# error a long way from its cause. Asserted, because the comment above predicted this exact
# failure while the image shipped it.
RUN test -w /work || { echo "/work is not writable by $(id -un); HF_HOME is unusable"; exit 1; }

ENTRYPOINT ["senbonzakura"]
CMD ["--help"]
