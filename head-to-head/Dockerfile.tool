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
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# PyTorch pinned to the same major the project requires (MIN_TORCH is 2.5). cu124 wheels match the
# driver on the card. Pinned exactly rather than floated: a benchmark whose torch version moves
# under it is not a benchmark.
RUN pip install --no-cache-dir \
      torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# numpy is not a torch dependency but torch warns loudly without it and several tools assume it.
# Pinned like everything else: a benchmark whose numeric stack moves under it is not a benchmark.
RUN pip install --no-cache-dir numpy==2.1.3

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
