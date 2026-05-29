#!/usr/bin/env bash
# R1.3 — CI-style smoke test for the c2mf reproduction pipeline.
#
# Verifies environment + dataloaders + core model imports in ~10-30 sec so
# the §4 sweep won't waste GPU time on a broken setup.
#
# Runs in two environments:
#   - Mac local:        bash scripts/c2mf_smoke_test.sh
#                       (auto-picks .venv/bin/python, data at ~/data/csc_datasets)
#   - DGX container:    docker exec c2mf-dev bash -lc 'cd /workspace && bash scripts/c2mf_smoke_test.sh'
#                       (auto-picks /data/csc_datasets bind-mount, default `python`)
#
# Phase: R1 — Upstream Baseline Reproduction (Tenali 2026).
# See paper/outline/R1-baseline-reproduction-plan.md for context.

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

PY="${PY:-python}"
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; fi

if [ -d "/data/csc_datasets" ]; then
    DATA_DIR="${DATA_DIR:-/data/csc_datasets}"
else
    DATA_DIR="${DATA_DIR:-${HOME}/data/csc_datasets}"
fi
export DATA_DIR
export FSDD_PATH="${FSDD_PATH:-${DATA_DIR}/fsdd/recordings}"

echo "[smoke] python   = ${PY}"
echo "[smoke] data_dir = ${DATA_DIR}"
echo "[smoke] fsdd_path= ${FSDD_PATH}"

"${PY}" - <<'PYEOF'
import os, sys, traceback

repo = os.getcwd()
sys.path.insert(0, repo)

# 1. packages wiring — must match the recipe §3 / main.py pattern so torchfsdd
#    + simple_einet etc. resolve via packages/__init__.py.
from packages import PACKAGE_DICT
for pkg, sub in PACKAGE_DICT.items():
    sys.path.append(os.path.join(repo, "packages", sub))

DATA_DIR = os.environ["DATA_DIR"]

# 2. FSDD audio split sizes (catches the packages/__init__.py + FSDD_PATH patches).
from dataloader.clean_avmnist.get_data import load_fsdd
a_train, a_val, a_test = load_fsdd()
assert (len(a_train), len(a_val), len(a_test)) == (2400, 300, 300), \
    f"FSDD split sizes mismatch: got {(len(a_train), len(a_val), len(a_test))}"
print(f"[smoke] FSDD ✓               (2400 / 300 / 300)")

# 3. Core model imports. Order matters: models.fusion FIRST so that the
#    full models/__init__.py finishes (including .predictor on line 4)
#    before einet.py is loaded standalone. Loading einet first triggers
#    layers/einsum.py → models.predictor → models/__init__.py → .fusion
#    → einet (mid-import) → circular. Recipe §3 has the wrong order.
from models.fusion import CredibilityWeightedMean                      # noqa: F401
from models.base import LateFusionClassifier                           # noqa: F401
from packages.EinsumNet.simple_einet.einet import Einet                # noqa: F401
print(f"[smoke] core imports ✓       (CredibilityWeightedMean, LateFusionClassifier, Einet)")

# 4. AVMNIST dataloader — 1 batch via the public get_dataloader entrypoint
#    that the experiment configs use. Mirrors the joint-training pattern.
from dataloader.clean_avmnist.get_data import get_dataloader as av_get_dataloader
av_train_dl, av_val_dl, av_test_dl = av_get_dataloader(
    data_dir=os.path.join(DATA_DIR, "clean_avmnist"),
    batch_size=4, num_workers=0,
    noise_severity=1, test_noise=0.5,
    exp_setup="both_modalities_one_twice_as_severe",
)
batch = next(iter(av_train_dl))
print(f"[smoke] AVMNIST 1-batch ✓    "
      f"(modalities={len(batch[0]) if isinstance(batch[0], (list, tuple)) else '?'}, "
      f"batch_size={batch[-1].shape[0] if hasattr(batch[-1], 'shape') else len(batch[-1])})")

# 5. NYUD dataloader — same 1-batch pattern. FINE_SIZE / LOAD_SIZE from
#    conf/experiment/nyud2_credibility_weighted.yaml defaults.
from dataloader.nyud2.get_data import get_dataloader as ny_get_dataloader
ny_train_dl, ny_val_dl, ny_test_dl = ny_get_dataloader(
    data_dir=os.path.join(DATA_DIR, "nyud2", "nyud2_trainvaltest"),
    FINE_SIZE=224, LOAD_SIZE=256,
    batch_size=4, num_workers=0, train_shuffle=False,
    noise_severity=1, test_noise=0.5,
    exp_setup="both_modalities_one_twice_as_severe",
)
batch = next(iter(ny_train_dl))
print(f"[smoke] NYUD 1-batch ✓       "
      f"(modalities={len(batch[0]) if isinstance(batch[0], (list, tuple)) else '?'}, "
      f"batch_size={batch[-1].shape[0] if hasattr(batch[-1], 'shape') else len(batch[-1])})")

print("[smoke] OK")
PYEOF
