# A sealed box for running somebody else's abliteration tool.
#
# WHY THIS EXISTS
#
# The benchmark compares senbonzakura against Heretic, and later against abliterix and
# OBLITERATUS. Comparing means RUNNING them, and each pulls a large dependency tree and executes
# arbitrary PyTorch. The card they would run on holds a HuggingFace token and ssh keys to a
# private hub. Baseline section 5 is explicit that untrusted third-party builds do not run as the
# user who holds the credentials, and a benchmark is not a good enough reason to make an exception.
#
# None of this implies the tools are hostile. They are AGPL research projects with real users and
# named authors. It implies that "probably fine" is not a control, and that a comparison published
# with an invitation for others to reproduce it should be reproducible without asking anyone to
# trust our judgement about somebody else's dependency tree.
#
# WHAT ISOLATION ACTUALLY BUYS HERE
#
# The container is the second line, not the first. The first is `--network none` at run time: a
# tool that cannot reach the network cannot exfiltrate a credential it never had. Everything the
# tool needs (weights, corpus) is mounted read-only, and the only writable path is its output
# directory. See `head-to-head/run-isolated.sh`, which refuses to start if those invariants are not met.
#
# WHY NOT AN NVIDIA BASE IMAGE
#
# On WSL2 the GPU arrives through `/dev/dxg` and the driver libraries in `/usr/lib/wsl/lib`, both
# bind-mounted at run time. PyTorch's own wheels carry the CUDA runtime they need and take only
# `libcuda.so` from the host. So the nvidia-container-toolkit is not installed and no CUDA base
# image is pulled: two large dependencies avoided, which is the point of section 5 rather than an
# optimisation.
#
# Build:  docker build -f head-to-head/Dockerfile.tool -t senbon-bench:tool .
FROM python:3.11-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

# git only, for cloning the tool under test at a pinned ref. No curl, no build toolchain, nothing
# that widens the surface inside a box whose whole purpose is to be narrow.
#
# THESE TWO ARE DELIBERATELY NOT PINNED, and baseline §5 says to pin what a lockfile does not
# manage, so the reasoning is here rather than left as an omission.
#
# An exact Debian pin (`git=1:2.39.5-0+deb12u2`) only keeps building while that exact version is
# in the archive, and a bookworm point release drops the superseded one from the main mirror. So
# the pin's real cost is that the image stops building on a day nobody chose, and buying back a
# reproducible build then means pointing apt at snapshot.debian.org, which is a second archive to
# trust inside a box whose whole argument is a narrow surface.
#
# What the pin would buy here is also small, in a way that is specific to these two packages:
#
#   * `git` is used once, to clone the tool under test at a ref, and the CLONE'S OWN COMMIT is
#     recorded into the image (see Dockerfile.heretic). A different git fetches the same commit or
#     fails; it cannot quietly fetch a different one.
#   * `ca-certificates` floating is the desirable direction. It is a trust store, and an old one
#     keeps trusting something that has been revoked.
#
# Neither package is importable by a tool and neither can change a number. The base image is
# pinned by DIGEST on the line above, which fixes the apt versions for any given build of this
# image anyway; what floats is only which versions a REBUILD picks up. The resolved set is
# recorded either way: `/opt/bench-env.txt` below carries the Python side, and `dpkg -l` inside the
# image answers the system side for anybody who needs it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# THE DECLARED ENVIRONMENT, AUTHORED ONCE AND CONSUMED THREE TIMES.
#
# These two `ARG`s are the single place this image's pins are written. They feed the `pip install`
# below, the image `LABEL` a host reads with `docker inspect`, the `ENV` a running arm reads from
# inside, and the build-time check that holds the image to them. Before this, the pins lived in the
# install command and the assertion repeated a literal, and a repeated literal is a thing that can
# disagree with itself.
ARG TORCH_PIN=2.5.1
ARG NUMPY_PIN=2.4.6
ARG BENCH_PINS="torch==$TORCH_PIN numpy==$NUMPY_PIN"
# Packages that must be importable whose version is NOT pinned here. Kept apart from the pins
# deliberately: asserting a version nobody chose would be a check wearing a pin's clothes. The
# child images extend this; `tokenizers` arrives with transformers and belongs to them.
ARG BENCH_REQUIRED=""

# PyTorch pinned to the same major the project requires (MIN_TORCH is 2.5). cu124 wheels match the
# driver on the card. Pinned exactly rather than floated: a benchmark whose torch version moves
# under it is not a benchmark.
RUN pip install --no-cache-dir \
      "torch==$TORCH_PIN" --index-url https://download.pytorch.org/whl/cu124

# numpy is not a torch dependency but torch warns loudly without it and several tools assume it.
# Pinned like everything else: a benchmark whose numeric stack moves under it is not a benchmark.
#
# THE PIN WAS `2.1.3` AND IT DID NOT HOLD. Recorded rather than quietly replaced, because the
# comment above was right for two months while the code under it was not.
#
# Heretic v1.4.0's `pyproject.toml` declares `numpy~=2.2`, which is `>=2.2, ==2.*`, and `2.1.3`
# does not satisfy it. `Dockerfile.heretic` installs the cloned tool with no `--no-deps` and no
# numpy constraint of its own, so pip MUST have upgraded numpy in that layer on every build ever
# made, while the senbonzakura image's pins left it at 2.1.3. **The two arms of a published
# comparison therefore differed in their numeric stack by construction**, and `CONTRACT.md`
# presents that comparison as one environment.
#
# Nothing caught it, and the reason is the useful part: both child images did assert a version, and
# both asserted `torch`, which was never the package that moved. A one-package assertion cannot
# see a second package move.
#
# `2.4.6` is the resolution, decided by the operator (Q-47) rather than left to pip. It satisfies
# Heretic's `~=2.2`, and the senbonzakura image's pinned transformers 5.14.1 and datasets 5.0.1
# declare only `numpy>=1.17`, so it satisfies that arm too. One explicit pin serves both and leaves
# neither tool below a declared floor. Both child images pin it again at this version: that is the
# point rather than redundancy, because an arm that inherits its numeric stack from somebody else's
# resolution cannot state it.
#
# IT IS NOT THE NEWEST 2.x, AND THE REASON IS THE CEILING ABOVE IT. This first said `2.5.3`, on the
# strength of that being the newest 2.x on PyPI, and the build refused it: **numpy 2.5.x declares
# `>=3.12` and this image's base is pinned by digest to `python:3.11-slim-bookworm`.** So the real
# ceiling here is 2.4.6, which is the highest 2.x with a Python 3.11 wheel, and "the newest 2.x"
# was never the operative question. Raising it past 2.4.6 means moving the base image's Python,
# which changes the interpreter under BOTH arms and is a larger decision than a numpy pin, so it
# is not taken here.
#
# Worth noticing how this was caught: by building, not by reading. The version existed, PyPI
# served its metadata, and every check in this file would have passed a declared string naming it.
# The only thing that knew it was unusable was pip, resolving against a real interpreter.
#
# Every published head-to-head arm is a re-measurement under this change. The 2026-09-10 results
# are superseded, not deleted; they stand as the record of what was measured under the old
# condition.
RUN pip install --no-cache-dir "numpy==$NUMPY_PIN"

# THE CHECK, AT THE WIDTH OF THE QUESTION. Reads the declared string rather than a literal, so the
# enforced set and the recorded set are the same string and extending it is editing `BENCH_PINS`.
# `head-to-head/benchenv.py` carries the comparison and the reasoning; it is copied in rather than
# inlined so the build check and the run-time record are one implementation.
#
# THE CHILD IMAGES INHERIT THIS COPY, so editing `benchenv.py` and rebuilding only a child leaves
# that child verifying itself with the baked older copy. Rebuild this base first. The blast radius
# is the build check alone: at RUN time `run-isolated.sh` mounts `head-to-head/` at `/work/bench`
# and the drivers import the live file from there, so an arm's recorded environment always comes
# from the checkout rather than from the image.
COPY head-to-head/benchenv.py /opt/benchenv.py
ENV BENCH_DECLARED_PINS=$BENCH_PINS \
    BENCH_REQUIRED_PRESENT=$BENCH_REQUIRED
RUN python /opt/benchenv.py --verify-pins

# THE RESOLVED SET, BAKED, so the environment can be read from the artefact rather than from an
# image somebody still happens to hold.
#
# `CONTRACT.md` promises each row carries its image digest, and a digest identifies an environment
# without describing one: it cannot be resolved back into a package list from this repository. The
# freeze can. A child image that installs anything RE-RUNS this line, because a freeze taken in
# the base describes the base and would be read as describing the arm.
#
# This is the resolved layer of three. The LABELs and ENV above carry what was DECLARED; this file
# carries what pip actually installed; `benchenv.py` reads what the running arm can actually
# import. They are different claims, and conflating them is how the numpy move stayed invisible.
RUN pip freeze > /opt/bench-env.txt \
 && echo "bench-env.txt: $(wc -l < /opt/bench-env.txt) distributions"

# From the same ARG as the ENV and the install, so a host reading labels and an arm reading its own
# environment cannot be told two different things.
LABEL org.senbonzakura.bench.declared-pins=$BENCH_PINS \
      org.senbonzakura.bench.required-present=$BENCH_REQUIRED \
      org.senbonzakura.bench.base-image="python:3.11-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3" \
      org.senbonzakura.bench.freeze="/opt/bench-env.txt"

# The tool under test is NOT baked in. It is cloned at run time at a ref the runner records, so
# the image does not have to be rebuilt to benchmark a different tool or a different version, and
# so the ref that produced a number is in the result rather than in an image tag.

# Nothing runs as root. The uid matches the host operator so mounted outputs are not root-owned,
# which otherwise needs a sudo to clean up and invites doing this as root instead.
ARG UID=1000
ARG GID=1000
RUN groupadd -g "$GID" bench && useradd -m -u "$UID" -g "$GID" bench
USER bench
WORKDIR /work

# The WSL driver libraries are bind-mounted here at run time.
ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib
# Fail loudly rather than silently reaching for a network that run-isolated.sh has removed.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/work/cache \
    PYTHONUNBUFFERED=1

CMD ["python", "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"]
