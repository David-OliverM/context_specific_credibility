# C²MF reproduction environment — TU-DA AIML DGX-2 (V100 / CUDA 11.7)
# Build: docker build -t c2mf:latest .
# Run (on DGX):
#   docker run -d -it --name c2mf-dev --gpus device=N \
#     -v $PWD:/workspace \
#     -v $HOME/data:/data \
#     c2mf:latest bash
#
# This Dockerfile is the canonical recipe for the C²MF dev environment.
# Anything below replaces the manual `pip install` steps used during initial
# DGX onboarding (2026-05-11) and the convenience snapshot image
# `c2mf-dev:2026-05-11` (kept as a fallback during the Modell-B migration).

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Avoid interactive tzdata prompts during apt-get install.
ENV DEBIAN_FRONTEND=noninteractive

# git + unzip are useful inside the container (the base image lacks both).
# wget kept around for ad-hoc dataset pulls.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        unzip \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# pkg_resources from setuptools >= 66 trips Python 3.10's distutils contract
# used by some of our dependencies. Pinning fixes pip resolution.
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
