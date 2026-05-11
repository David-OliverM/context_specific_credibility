#!/usr/bin/env bash
# Container entrypoint — vendors torchfsdd lib into the bind-mounted workspace
# if it isn't already there. No-op on subsequent runs.
set -e

LIB_DIR="/workspace/packages/torchfsdd/lib/torchfsdd"
STAGE="/opt/torchfsdd-stage/torch-fsdd-1.0.0/lib/torchfsdd"

if [ -d "/workspace/packages/torchfsdd" ] && [ ! -d "$LIB_DIR" ] && [ -d "$STAGE" ]; then
    mkdir -p "$LIB_DIR"
    cp -r "$STAGE/." "$LIB_DIR/"
    echo "[c2mf-entrypoint] vendored torchfsdd lib into $LIB_DIR"
fi

exec "$@"
