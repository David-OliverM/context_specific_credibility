"""Frankfurt fMRI dataloader for context-specific-credibility.

Loads the 135-scan Frankfurt-Neurodata-UKF dataset (45 subjects x 3 conditions)
and returns per-modality functional-connectivity (FC) feature vectors plus the
drug label.

Key facts (verified at /Users/david-oliver.matzka/dev/research-C2MF/paper/notebooks/01_explore_frankfurt_neurodata.ipynb):
- 132 ROIs (Harvard-Oxford + AAL cerebellum); ROI label file shipped alongside CSVs.
- 267 timesteps per scan; 2 outliers (LDopa/Subject001, Placebo/Subject003) at
  268 -> truncated to 267.
- Values mean-centered per column, std ~ 0.20-0.32 (NOT z-scored).
- 3 classes: Amisulpride / LDopa / Placebo.
- Subject-wise split MANDATORY (random splits collapse to chance).

The data layout from `data_dir`:
    Amisulpride/ROI_Subject{001..045}_Condition001_atlas_timeseries.csv
    LDopa/      ROI_Subject{001..045}_Condition002_atlas_timeseries.csv
    Placebo/    ROI_Subject{001..045}_Condition003_atlas_timeseries.csv
    ROI_labels.txt

Each item yields the same triple-of-tuples shape that `LateFusionClassifier`
expects (mirroring `clean_avmnist`):
    ((mod_1_features, ..., mod_M_features, label),
     (idx, sample_corr, corr_modalities),
     (mod_1_noise_mask, ..., mod_M_noise_mask))

The "noise" channel is unused for the Frankfurt task (no synthetic
modality-corruption defined yet) -> we yield zero tensors so the
`noise_encoders` path in models/base.py stays well-defined.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

# Make sure the repo root is on sys.path so we can import host-repo modules if
# we ever need to (mirrors the clean_avmnist pattern).
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONDITIONS = ("Amisulpride", "LDopa", "Placebo")
LABEL_MAP = {c: i for i, c in enumerate(CONDITIONS)}
N_TIMESTEPS = 267  # truncate everything to this length

SUBCORTICAL_PATS = re.compile(
    r"^(Thalamus|Caudate|Putamen|Pallidum|Hippocampus|Amygdala|Accumbens|Brain-Stem)",
    re.IGNORECASE,
)
CEREBELLAR_PATS = re.compile(r"^(Cereb|Ver)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# ROI-label parsing  (taken from paper/scripts/baselines_frankfurt_late_fusion.py)
# ---------------------------------------------------------------------------

def _parse_roi_labels(roi_labels_path: Path) -> list[str]:
    """Parse `ROI_labels.txt` into short tokens like 'FP_R', 'Thalamus_R'."""
    short = []
    with open(roi_labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'"atlas\.([^("]+)\s*(?:\(.*)?"$', line)
            if not m:
                # Fallback: keep the raw bit between quotes.
                short.append(line.strip('"').replace("atlas.", "").strip())
                continue
            token = m.group(1).strip()
            token = re.sub(r"\s+([rRlL])$", lambda m: f"_{m.group(1).upper()}", token)
            short.append(token.replace(" ", ""))
    return short


# ---------------------------------------------------------------------------
# Modality groupings
# ---------------------------------------------------------------------------

def _anatomical_grouping(roi_labels: Sequence[str]) -> dict[str, list[int]]:
    """3-way: cortical / subcortical / cerebellar.  Derivable from ROI names."""
    out: dict[str, list[int]] = {"cortical": [], "subcortical": [], "cerebellar": []}
    for i, lab in enumerate(roi_labels):
        if SUBCORTICAL_PATS.match(lab):
            out["subcortical"].append(i)
        elif CEREBELLAR_PATS.match(lab):
            out["cerebellar"].append(i)
        else:
            out["cortical"].append(i)
    return out


# Bucket order for the dopamine-circuit grouping.  Lorina Zapf (UKF) defined
# four primary buckets plus an `other` catch-all so no ROI is silently dropped.
_DOPAMINE_BUCKET_ORDER = (
    "dorsal_striatum", "ventral_striatum", "midbrain_proxy",
    "da_projection_cortex", "other",
)

_DOPAMINE_CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "conf" / "grouping" / "dopamine_circuit_v2.csv"
)


# Bucket order matches Yeo 2011 network ordering, with two auxiliary trailing
# buckets for ROIs outside the cortical Yeo-7 definition (subcortex + cerebellum).
_YEO7_BUCKET_ORDER = (
    "VIS", "SOMMOT", "DORSATTN", "VENTATTN", "LIMBIC", "CONT", "DEFAULT",
    "SUBCORTICAL", "CEREBELLAR",
)

_YEO7_CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "conf" / "grouping" / "yeo7_frankfurt_v2_symmetric.csv"
)


def _load_bucket_csv(csv_path: Path, bucket_column: str) -> dict[int, str]:
    """Returns {roi_idx: bucket_label}, parsed from a mapping CSV.

    Generic loader used by both _yeo7_mapping and _dopamine_grouping.
    Expects a header line with 'roi_idx' and the requested bucket_column.
    """
    out: dict[int, str] = {}
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        idx_col = header.index("roi_idx")
        b_col = header.index(bucket_column)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            out[int(parts[idx_col])] = parts[b_col].strip()
    return out


def _grouping_from_csv(roi_labels: Sequence[str], csv_path: Path,
                      bucket_order: tuple, bucket_column: str,
                      grouping_name: str, log_message: str) -> dict[str, list[int]]:
    """Generic CSV-driven grouping builder.  Validates ROI count, ensures
    every ROI is assigned a known bucket, drops buckets with k<2.
    """
    if len(roi_labels) != 132:
        raise ValueError(
            f"{grouping_name} mapping expects 132 ROIs (Frankfurt atlas); "
            f"got {len(roi_labels)}. The mapping CSV would need to be re-derived "
            "for a different ROI set."
        )
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{grouping_name} mapping CSV not found at {csv_path}. "
            "Cannot proceed without explicit mapping (refuse silent fallback)."
        )
    print(f"[frankfurt] {log_message}")
    idx_to_bucket = _load_bucket_csv(csv_path, bucket_column)
    out: dict[str, list[int]] = {b: [] for b in bucket_order}
    for i in range(len(roi_labels)):
        bucket = idx_to_bucket.get(i)
        if bucket is None:
            raise ValueError(
                f"ROI idx {i} ({roi_labels[i]}) missing from {grouping_name} CSV "
                f"({csv_path.name})"
            )
        if bucket not in out:
            raise ValueError(
                f"ROI idx {i} ({roi_labels[i]}) has unknown bucket '{bucket}'; "
                f"expected one of {bucket_order}"
            )
        out[bucket].append(i)
    # Drop buckets with <2 ROIs (k<2 -> empty FC upper triangle).
    return {k: v for k, v in out.items() if len(v) >= 2}


def _yeo7_mapping(roi_labels: Sequence[str]) -> dict[str, list[int]]:
    """Yeo-7 functional networks (Yeo et al. 2011, Cerebral Cortex).

    Loads code/conf/grouping/yeo7_frankfurt_v2_symmetric.csv: a per-ROI
    assignment derived by overlaying nilearn's Yeo-2011 7-network MNI152
    template on the Harvard-Oxford cortical labelmap (cort-maxprob-thr25-2mm)
    and taking the modal Yeo label per ROI mask, with L+R pairs constrained
    to share a bucket (combined-voxel-argmax).  Subcortical (15) and
    cerebellar+vermis (26) ROIs are routed into auxiliary SUBCORTICAL and
    CEREBELLAR buckets since Yeo-7 is not defined off-cortex.

    Provenance: paper/scripts/verify_yeo7_mapping.py.
    Audit trail: code/conf/grouping/yeo7_v1_vs_v2_diff.md compares this v2
    mapping against the prior v1 heuristic.
    """
    return _grouping_from_csv(
        roi_labels, _YEO7_CSV_PATH, _YEO7_BUCKET_ORDER,
        bucket_column="yeo7_network",
        grouping_name="Yeo-7",
        log_message=(
            f"Yeo-7 mapping loaded from {_YEO7_CSV_PATH.name} "
            "(MNI-overlay verified, L=R symmetric). "
            "See code/conf/grouping/README.md for provenance."
        ),
    )


def _dopamine_grouping(roi_labels: Sequence[str]) -> dict[str, list[int]]:
    """Dopamine-circuit grouping after Lorina Zapf (UKF, 2026-05-07).

    Loads code/conf/grouping/dopamine_circuit_v2.csv: a per-ROI assignment
    of the 132 Frankfurt ROIs into five buckets representing the canonical
    cortico-striato-mesencephalic dopamine circuit:

      - dorsal_striatum      : Caudate, Putamen (L+R) - primary D2/D3 sites
      - ventral_striatum     : Accumbens (L+R)        - mesolimbic reward
      - midbrain_proxy       : Brain-Stem (single ROI) - VTA/SN proxy; falls
                               through the k>=2 filter and is dropped at
                               training time, leaving M=4 effective
      - da_projection_cortex : vmPFC, OFC, ACC, Amygdala, dlPFC components
                               (13 cortical+subcortical ROIs)
      - other                : every ROI not in Lorina's explicit circuit
                               (112 ROIs); kept so no data is silently lost

    Provenance: thesis/correspondence/2026-05-07_zapf_data-questions.pdf
    plus the v1 heuristic that preceded Lorina's reply (kept in the audit
    trail at code/conf/grouping/README.md).
    """
    return _grouping_from_csv(
        roi_labels, _DOPAMINE_CSV_PATH, _DOPAMINE_BUCKET_ORDER,
        bucket_column="dopamine_bucket",
        grouping_name="Dopamine-Circuit",
        log_message=(
            f"Dopamine-circuit mapping loaded from {_DOPAMINE_CSV_PATH.name} "
            "(Lorina Zapf 2026-05-07; midbrain_proxy k=1 dropped). "
            "See code/conf/grouping/README.md for provenance."
        ),
    )


def _dopamine_grouping_M3(roi_labels: Sequence[str]) -> dict[str, list[int]]:
    """Dopamine grouping with the catch-all `other` bucket (112 ROIs) dropped.

    Diagnostic variant for F1.4c [[Decision Log - F1.5-F1.6 Sequence Inversion]]:
    removes the 1792-SAX-feature `other` bucket to test whether it is the
    primary overfit driver in F1.4b.  M=3 effective modalities:
    dorsal_striatum, ventral_striatum, da_projection_cortex.
    """
    out = _dopamine_grouping(roi_labels)
    out.pop("other", None)
    print("[frankfurt] dopamine_M3 grouping: dropped `other` bucket "
          f"({list(out.keys())} remaining)")
    return out


GROUPING_FNS = {
    "anatomical": _anatomical_grouping,
    "dopamine": _dopamine_grouping,
    "dopamine_M3": _dopamine_grouping_M3,
    "yeo7": _yeo7_mapping,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _enumerate_csvs(data_dir: Path) -> list[tuple[Path, str, int]]:
    """Returns [(path, condition, subject_id), ...]."""
    out: list[tuple[Path, str, int]] = []
    for cond in CONDITIONS:
        files = sorted((data_dir / cond).glob("ROI_Subject*_Condition*.csv"))
        for p in files:
            m = re.search(r"Subject(\d+)_Condition", p.name)
            assert m, f"Cannot parse subject id from {p.name}"
            out.append((p, cond, int(m.group(1))))
    return out


def _load_timeseries(p: Path) -> np.ndarray:
    """Load CSV and truncate to N_TIMESTEPS rows. Returns (T, 132) float32."""
    df = pd.read_csv(p)
    arr = df.values.astype(np.float32)
    if arr.shape[0] >= N_TIMESTEPS:
        arr = arr[:N_TIMESTEPS]
    else:
        raise ValueError(f"{p} has only {arr.shape[0]} timesteps (<{N_TIMESTEPS})")
    return arr


def _fc_upper(mat: np.ndarray) -> np.ndarray:
    """Upper-triangular off-diagonal entries of a square corr matrix (k*(k-1)/2,)."""
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu].astype(np.float32)


def _build_modality_features(ts: np.ndarray, modalities: dict[str, list[int]]) -> dict[str, np.ndarray]:
    """For each modality, compute the within-modality FC upper triangle."""
    out = {}
    for name, idxs in modalities.items():
        if len(idxs) < 2:
            # k=1 -> no off-diagonal; create zero-dim feature so downstream
            # code can still index the modality consistently.
            out[name] = np.zeros((0,), dtype=np.float32)
            continue
        sub = ts[:, idxs]                         # (T, k)
        # corrcoef expects (vars, obs) -> transpose
        fc = np.corrcoef(sub, rowvar=False)       # (k, k)
        # Replace NaNs (from constant rows) with 0
        fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
        out[name] = _fc_upper(fc)
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FrankfurtFCDataset(Dataset):
    """In-memory per-modality dataset for the Frankfurt fMRI task.

    Two output modes (selected at construction time):

    * ``emit_timeseries=False`` (default, used by the MLPEncoder pipeline):
      per modality returns the within-modality FC upper-triangular vector of
      shape ``(k*(k-1)/2,)``.  This is the path used by every config that has
      been on the SoT baseline tables (v1 50-run, E2 dropout05, E3 wd1e-2,
      etc.) — code-identical to the pre-F1.2 behaviour.

    * ``emit_timeseries=True`` (used by the F1 TabPFNSAXEncoder pipeline):
      per modality returns the raw ROI time-series slice of shape
      ``(N_TIMESTEPS, k_rois)``.  Memory footprint ~19 MB for the full
      135-scan cache.

    Both modes share the same triple-of-tuples ``__getitem__`` shape so the
    downstream FusionModel / training loop can stay agnostic; only the
    per-modality tensor content changes.
    """

    def __init__(
        self,
        items: list[tuple[Path, str, int]],
        modalities: dict[str, list[int]],
        modality_order: list[str],
        emit_timeseries: bool = False,
    ):
        self.items = items
        self.modalities = modalities
        self.modality_order = modality_order
        self.emit_timeseries = bool(emit_timeseries)
        # Cache stores per-modality numpy arrays (FC-vec OR (T, k_rois) slice).
        self.cache: list[tuple[list[np.ndarray], int, int]] = []
        for p, cond, subj in items:
            ts = _load_timeseries(p)                       # (T, 132)
            if self.emit_timeseries:
                per_mod = [
                    ts[:, modalities[m]].astype(np.float32)
                    for m in modality_order
                ]
            else:
                feats_by_mod = _build_modality_features(ts, modalities)
                per_mod = [feats_by_mod[m] for m in modality_order]
            label = LABEL_MAP[cond]
            self.cache.append((per_mod, label, subj))

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx: int):
        per_mod, label, _subj = self.cache[idx]
        feat_tensors = [torch.from_numpy(f).float() for f in per_mod]
        # noise masks: zeros, same shape as features (used by frozen
        # noise_encoders in models/base.py FusionModel).
        noise_masks = tuple(torch.zeros_like(t) for t in feat_tensors)
        # batch tuple: (mod_1, mod_2, ..., mod_M, label)
        batch_data = tuple(feat_tensors) + (torch.tensor(label, dtype=torch.long),)
        sample_corr = ["none"] * len(feat_tensors)
        corr_modalities = torch.zeros(len(feat_tensors), dtype=torch.bool)
        return batch_data, (idx, sample_corr, corr_modalities), noise_masks


# ---------------------------------------------------------------------------
# Splitting (subject-wise GroupKFold)
# ---------------------------------------------------------------------------

def _subject_kfold(items: list[tuple[Path, str, int]], n_splits: int, fold: int):
    """Return (train_items, val_items, test_items).  Val and test split the
    held-out fold's subjects 50/50 (small data, so we trade test-size for
    val-set existence)."""
    subjects = np.array([s for _, _, s in items])
    indices = np.arange(len(items))
    # GroupKFold deterministic ordering -> picks the fold-th split.
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(indices, groups=subjects))
    train_idx, holdout_idx = splits[fold]
    holdout_subjects = np.unique(subjects[holdout_idx])
    rng = np.random.default_rng(42 + fold)
    rng.shuffle(holdout_subjects)
    half = max(1, len(holdout_subjects) // 2)
    val_subjects = set(holdout_subjects[:half].tolist())
    test_subjects = set(holdout_subjects[half:].tolist())
    val_idx = [i for i in holdout_idx if subjects[i] in val_subjects]
    test_idx = [i for i in holdout_idx if subjects[i] in test_subjects]
    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    test_items = [items[i] for i in test_idx]
    return train_items, val_items, test_items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_modality_dims(data_dir: str, modality_grouping: str = "anatomical"):
    """Helper for config introspection: returns ordered (name, in_dim) tuples."""
    data_path = Path(data_dir)
    roi_labels = _parse_roi_labels(data_path / "ROI_labels.txt")
    grouping_fn = GROUPING_FNS[modality_grouping]
    modalities = grouping_fn(roi_labels)
    out = []
    for name, idxs in modalities.items():
        k = len(idxs)
        in_dim = k * (k - 1) // 2 if k >= 2 else 0
        out.append((name, in_dim))
    return out


def get_dataloader(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 0,
    noise_severity=None,        # accepted for API parity; unused
    test_noise: float = 0.0,    # accepted for API parity; unused
    exp_setup: str = "both_modalities_one_twice_as_severe",  # unused
    modality_grouping: str = "anatomical",
    n_splits: int = 5,
    fold: int = 0,
    emit_timeseries: bool = False,
    **kwargs,
):
    """Return (train_loader, val_loader, test_loader) for Frankfurt fMRI.

    Args:
      data_dir: path containing Amisulpride/, LDopa/, Placebo/, ROI_labels.txt.
      modality_grouping: 'anatomical' | 'dopamine' | 'yeo7'.
      n_splits, fold: subject-wise GroupKFold params.
      emit_timeseries: if True, per-modality output becomes the raw ROI
        time-series slice (shape ``(N_TIMESTEPS, k_rois)`` per sample);
        if False (default), output is the FC upper-triangular feature vector
        (shape ``(k*(k-1)/2,)``).  TabPFNSAXEncoder uses ``True``; the
        existing MLPEncoder pipeline uses ``False``.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Frankfurt data_dir does not exist: {data_path}")
    roi_labels = _parse_roi_labels(data_path / "ROI_labels.txt")

    if modality_grouping not in GROUPING_FNS:
        raise ValueError(
            f"Unknown modality_grouping={modality_grouping}. "
            f"Choose one of {list(GROUPING_FNS)}"
        )
    grouping_fn = GROUPING_FNS[modality_grouping]
    modalities = grouping_fn(roi_labels)
    modality_order = list(modalities.keys())

    # Print a short summary so the user knows what they're getting.
    print(f"[frankfurt] modality_grouping={modality_grouping}  "
          f"modalities={[(m, len(modalities[m])) for m in modality_order]}")
    for m in modality_order:
        k = len(modalities[m])
        in_dim = k * (k - 1) // 2 if k >= 2 else 0
        print(f"[frankfurt]   {m:>22s}: k={k:>3d}  in_dim={in_dim}")

    items = _enumerate_csvs(data_path)
    print(f"[frankfurt] loaded {len(items)} CSVs from {data_path}")
    assert len(items) == 135, f"expected 135 scans, got {len(items)}"

    train_items, val_items, test_items = _subject_kfold(items, n_splits, fold)
    print(f"[frankfurt] split (fold {fold}/{n_splits}): "
          f"train={len(train_items)}  val={len(val_items)}  test={len(test_items)}")

    train_set = FrankfurtFCDataset(
        train_items, modalities, modality_order,
        emit_timeseries=emit_timeseries,
    )
    val_set = FrankfurtFCDataset(
        val_items, modalities, modality_order,
        emit_timeseries=emit_timeseries,
    )
    test_set = FrankfurtFCDataset(
        test_items, modalities, modality_order,
        emit_timeseries=emit_timeseries,
    )
    if emit_timeseries:
        print(f"[frankfurt] emit_timeseries=True  -> per-modality outputs are "
              f"(T={N_TIMESTEPS}, k_rois) ROI time-series slices.")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=os.path.expanduser("~/dev/research-C2MF/paper/neuro_data_UKF"))
    p.add_argument("--grouping", default="anatomical")
    p.add_argument("--emit_timeseries", action="store_true",
                   help="F1.2: emit per-modality (T, k_rois) time-series "
                        "instead of FC features.")
    args = p.parse_args()
    tl, vl, te = get_dataloader(
        args.data_dir,
        batch_size=4,
        modality_grouping=args.grouping,
        emit_timeseries=args.emit_timeseries,
    )
    for batch_data, _, _ in tl:
        for i, t in enumerate(batch_data[:-1]):
            print(f"mod{i}: {tuple(t.shape)}  dtype={t.dtype}")
        print(f"label: {tuple(batch_data[-1].shape)}, sample={batch_data[-1].tolist()}")
        break
