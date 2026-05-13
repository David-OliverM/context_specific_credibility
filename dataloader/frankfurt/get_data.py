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


def _dopamine_grouping(roi_labels: Sequence[str]) -> dict[str, list[int]]:
    """Domain-motivated dopamine-circuit grouping (Lab recommendation).

    M = 4 buckets:
      - dorsal_striatum      : Caudate / Putamen / Pallidum (L+R)
      - ventral_striatum     : Accumbens (L+R)
      - midbrain_proxy       : Brain-Stem  (NOTE: Harvard-Oxford has no clean
                               VTA/SN ROI; Brain-Stem is the closest proxy)
      - da_projection_cortex : vmPFC / OFC / ACC / dlPFC heuristic
                               (matches Frontal Medial Cortex, Frontal Pole,
                                Frontal Orbital Cortex, Cingulate Anterior,
                                Superior+Middle Frontal Gyrus, ParaCing).

    Anything that doesn't match goes into a 5th `other` bucket so we never
    drop ROIs silently.  Only non-empty buckets become modalities.
    """
    DORSAL = re.compile(r"^(Caudate|Putamen|Pallidum)", re.IGNORECASE)
    VENTRAL = re.compile(r"^(Accumbens)", re.IGNORECASE)
    MIDBRAIN = re.compile(r"^(Brain-Stem)", re.IGNORECASE)
    # DA-projection cortex heuristic - matches Harvard-Oxford label tokens.
    PROJECTION = re.compile(
        r"^(MedFC|FP|FOrb|aCG|pCG|SubCalC|PaCiG|SFG|MidFG|FrOper)",
        re.IGNORECASE,
    )

    out: dict[str, list[int]] = {
        "dorsal_striatum": [],
        "ventral_striatum": [],
        "midbrain_proxy": [],
        "da_projection_cortex": [],
        "other": [],
    }
    for i, lab in enumerate(roi_labels):
        if DORSAL.match(lab):
            out["dorsal_striatum"].append(i)
        elif VENTRAL.match(lab):
            out["ventral_striatum"].append(i)
        elif MIDBRAIN.match(lab):
            out["midbrain_proxy"].append(i)
        elif PROJECTION.match(lab):
            out["da_projection_cortex"].append(i)
        else:
            out["other"].append(i)
    # Drop buckets with <2 ROIs (k<2 -> empty FC upper triangle, breaks MLP).
    return {k: v for k, v in out.items() if len(v) >= 2}


def _yeo7_mapping(roi_labels: Sequence[str]) -> dict[str, list[int]]:
    """Yeo-7 functional networks (Yeo et al. 2011, Cerebral Cortex).

    TODO(yeo7): The mapping below is a STUB.  A proper implementation requires
    the Yeo-7 network assignment per Harvard-Oxford+AAL ROI, which is not
    derivable from ROI names alone -- it has to be looked up in
    Yeo et al. 2011 (Cerebral Cortex) Tables 1 / S1, OR derived by overlaying
    the atlases in MNI space (e.g. via nilearn) and taking the modal Yeo-7
    label per ROI.

    For now this falls back to the anatomical 3-way grouping so the code
    runs end-to-end; the user must replace this with the real mapping
    before any Yeo-7 results are reported.
    """
    # FALLBACK / STUB.  Replace before reporting any "Yeo-7" results!
    return _anatomical_grouping(roi_labels)


GROUPING_FNS = {
    "anatomical": _anatomical_grouping,
    "dopamine": _dopamine_grouping,
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
    """In-memory FC-feature dataset (small enough that this is fine: 135 scans)."""

    def __init__(
        self,
        items: list[tuple[Path, str, int]],
        modalities: dict[str, list[int]],
        modality_order: list[str],
    ):
        self.items = items
        self.modalities = modalities
        self.modality_order = modality_order
        self.cache: list[tuple[list[np.ndarray], int, int]] = []
        for p, cond, subj in items:
            ts = _load_timeseries(p)
            feats_by_mod = _build_modality_features(ts, modalities)
            feats = [feats_by_mod[m] for m in modality_order]
            label = LABEL_MAP[cond]
            self.cache.append((feats, label, subj))

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx: int):
        feats, label, _subj = self.cache[idx]
        feat_tensors = [torch.from_numpy(f).float() for f in feats]
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
    **kwargs,
):
    """Return (train_loader, val_loader, test_loader) for Frankfurt fMRI.

    Args:
      data_dir: path containing Amisulpride/, LDopa/, Placebo/, ROI_labels.txt.
      modality_grouping: 'anatomical' | 'dopamine' | 'yeo7' (yeo7 is a STUB).
      n_splits, fold: subject-wise GroupKFold params.
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

    train_set = FrankfurtFCDataset(train_items, modalities, modality_order)
    val_set = FrankfurtFCDataset(val_items, modalities, modality_order)
    test_set = FrankfurtFCDataset(test_items, modalities, modality_order)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=os.path.expanduser("~/dev/research-C2MF/paper/neuro_data_UKF"))
    p.add_argument("--grouping", default="anatomical")
    args = p.parse_args()
    tl, vl, te = get_dataloader(args.data_dir, batch_size=4, modality_grouping=args.grouping)
    for batch_data, _, _ in tl:
        for i, t in enumerate(batch_data[:-1]):
            print(f"mod{i}: {t.shape}")
        print(f"label: {batch_data[-1].shape}, sample={batch_data[-1].tolist()}")
        break
