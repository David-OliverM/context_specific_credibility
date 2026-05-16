# C²MF reproduction environment — TU-DA AIML DGX (V100 / CUDA 12.4)
# Build: docker build -t c2mf:latest .
# Run (on DGX, single-GPU default per CLUSTER.md §3):
#   docker run -d -it --name c2mf-dev \
#       --device nvidia.com/gpu=N \
#       --shm-size=8g --cpus=6 --cpuset-cpus="0-5,48-53" \
#       --restart=unless-stopped \
#       -v $HOME/dev/c2mf-git:/workspace \
#       -v $HOME/dev/research-C2MF/paper:/workspace_paper \
#       -v $HOME/dev/tabpfn-cache:/root/.cache \
#       -v $HOME/data:/data \
#       c2mf:latest sleep infinity
#
# Multi-GPU (only after Mattermost announcement in #please-give-more-gpu,
# per CLUSTER.md §3): repeat --device nvidia.com/gpu=0 ... --device nvidia.com/gpu=7
# and bump --cpus=24 --cpuset-cpus="0-23,48-71" (NUMA-Node 0).
#
# Changelog vs pre-2026-05-15 image (c2mf:latest old):
#   - base bumped pytorch:2.0.1-cu117 -> 2.5.1-cu124 (V100 sm_70 supported)
#   - tabpfn 2.2.1 + pyts 0.13.0 + optuna 4.8.0 added (F1.x foundation-model
#     pipeline encoder + HP-search infrastructure)
#   - numpy pinned <2 (scipy 1.11.4 not numpy-2 compatible)
#   - TabPFN cache volume-mount documented (persistence across docker rm)
#   - paper-repo bind-mount slot documented (Tech-Debt #3 prep)

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Avoid interactive tzdata prompts during apt-get install.
ENV DEBIAN_FRONTEND=noninteractive

# git + unzip useful inside the container (base image lacks both).
# wget kept around for ad-hoc dataset pulls.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        unzip \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel \
    && pip install "setuptools<66"

# Project dependencies. requirements.txt is COPY'd first to leverage Docker's
# layer cache: if only source code changes, this expensive layer is reused.
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && pip install gdown

# Vendored torchfsdd lib (unmodified eonu/torch-fsdd v1.0.0).
# Upstream `.gitignore` excludes `lib/` so we cannot commit it to the repo;
# fetching at build time keeps the fork lean and the recipe explicit.
RUN mkdir -p /opt/torchfsdd-stage \
    && cd /opt/torchfsdd-stage \
    && wget -q https://github.com/eonu/torch-fsdd/archive/refs/tags/v1.0.0.tar.gz \
    && tar -xzf v1.0.0.tar.gz \
    && rm v1.0.0.tar.gz

# Default FSDD path — overridable at `docker run -e FSDD_PATH=...` time.
ENV FSDD_PATH=/data/csc_datasets/fsdd/recordings

WORKDIR /workspace

# Entrypoint script lives in the repo at docker/c2mf-entrypoint.sh — see that
# file for what it does. Kept as a separate file so this Dockerfile builds
# under the legacy builder (DGX rootless Docker has no buildx).
COPY docker/c2mf-entrypoint.sh /usr/local/bin/c2mf-entrypoint.sh
RUN chmod +x /usr/local/bin/c2mf-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/c2mf-entrypoint.sh"]
CMD ["bash"]
