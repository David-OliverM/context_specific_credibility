"""ABIDE fMRI dataloader for context-specific-credibility (C²MF over Yeo-7 networks).

External-validation companion to the Frankfurt loader. Loads the ABIDE I rois_ho
ROI time-series (Harvard-Oxford, 111 ROIs) via nilearn `fetch_abide_pcp`, groups the
ROIs into the 7 Yeo-2011 functional networks (+ a SUBCORTICAL bucket) as C²MF MODALITIES,
and returns per-modality functional-connectivity feature vectors plus the ASD/TC label.

Why this exists (paper backlog N-YEO7): on ABIDE the per-network credibility-fusion pre-test
(paper/scripts/baselines_abide_yeo7_c2mf.py) showed that the EDGE-PRESERVING source definition
("rowblock": each network's connectivity to the WHOLE brain) lets multi-source fusion beat the
flat full-FC TabPFN baseline (0.739 vs 0.720 AUROC), while the within-network FC source loses
(drops cross-network edges, like the Frankfurt dopamine sub-circuits). The fixed entropy-
credibility rule only TIED plain mean-fusion, so the open question for the real C²MF is whether
the LEARNED per-instance credibility head beats naive mean-fusion. This loader feeds that test.

Source definition (`source_mode`):
  - rowblock (default, the pre-test winner): modality i = the rows of the full FC matrix for
    network i's ROIs, flattened -> (k_i * n_roi,). Keeps cross-network edges.
  - within: modality i = within-network FC upper-triangle -> (k_i*(k_i-1)/2,). Drops cross edges.

CV (`cv`):
  - subject (default): subject-wise GroupKFold(n_splits); fold k. (ABIDE is cross-sectional, so
    subject == scan, but we keep SUB_ID groups for parity / repeated scans.)
  - site: leave-one-site-out. `fold` indexes the sorted site list; that site is the test set.
    THE literature-standard ABIDE confound control (unseen scanner).

Each item yields the same triple-of-tuples as the Frankfurt FC path (emit_timeseries=False):
    ((mod_1, ..., mod_M, label), (idx, sample_corr, corr_modalities), (noise_mask_1, ...))
so the FusionModel / MLPEncoder / CredibilityWeightedMean pipeline stays agnostic.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

_YEO7_CSV = (Path(__file__).resolve().parent.parent.parent
             / "conf" / "grouping" / "abide_yeo7.csv")
_MIN_T = 30          # require at least this many timesteps (FC stability)


# ---------------------------------------------------------------------------
# Data loading (nilearn ABIDE, cached at data_dir)
# ---------------------------------------------------------------------------

def _load_abide(data_dir: str):
    """Return ts_list (list of (T_i, 111)), y (0=TC,1=ASD), subjects, sites."""
    from nilearn import datasets as nl_datasets
    abide = nl_datasets.fetch_abide_pcp(
        data_dir=data_dir, pipeline="cpac", band_pass_filtering=True,
        global_signal_regression=False, derivatives=["rois_ho"], quality_checked=True,
    )
    ts_all = abide["rois_ho"]
    pheno = abide["phenotypic"]
    dx = np.asarray(pheno["DX_GROUP"]).astype(int)      # 1=ASD, 2=control
    sub = np.asarray(pheno["SUB_ID"]).astype(int)
    site = np.asarray(pheno["SITE_ID"]).astype(str)
    ts_list, y, subjects, sites = [], [], [], []
    for i, ts in enumerate(ts_all):
        ts = np.asarray(ts, dtype=np.float64)
        if ts.shape[0] < _MIN_T:
            continue
        ts_list.append(ts)
        y.append(1 if dx[i] == 1 else 0)
        subjects.append(int(sub[i])); sites.append(str(site[i]))
    print(f"[abide] scans={len(ts_list)} ASD={int(np.sum(y))} TC={int(len(y)-np.sum(y))} "
          f"ROIs={ts_list[0].shape[1]} sites={len(set(sites))}", flush=True)
    return ts_list, np.asarray(y), np.asarray(subjects), np.asarray(sites)


# ---------------------------------------------------------------------------
# Yeo-7 grouping (positional: column index -> network) + modality features
# ---------------------------------------------------------------------------

def _yeo7_groups(n_roi: int) -> dict[str, list[int]]:
    """Read abide_yeo7.csv -> {network: [column positions]}, k>=2 only, stable order."""
    df = pd.read_csv(_YEO7_CSV)
    assert len(df) == n_roi, f"mapping rows {len(df)} != ROIs {n_roi}"
    groups: dict[str, list[int]] = {}
    for _, r in df.iterrows():
        groups.setdefault(str(r["group"]), []).append(int(r["pos"]))
    groups = {g: pos for g, pos in groups.items() if len(pos) >= 2}
    return {g: groups[g] for g in sorted(groups)}      # stable (alphabetical) order


def _modality_features(ts: np.ndarray, groups: dict[str, list[int]],
                       order: list[str], source_mode: str) -> list[np.ndarray]:
    """Per-modality FC features for one scan. cc = full (n_roi, n_roi) Pearson FC."""
    cc = np.nan_to_num(np.corrcoef(ts, rowvar=False), nan=0.0, posinf=0.0, neginf=0.0)
    out = []
    for g in order:
        pos = groups[g]
        if source_mode == "within":
            sub = cc[np.ix_(pos, pos)]
            iu = np.triu_indices(sub.shape[0], k=1)
            out.append(sub[iu].astype(np.float32))
        else:                                          # rowblock: network rows x all ROIs
            out.append(cc[pos, :].reshape(-1).astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AbideFCDataset(Dataset):
    """Per-modality (Yeo-7 network) FC dataset. Same triple-of-tuples as Frankfurt FC."""

    def __init__(self, ts_list, y, groups, order, source_mode):
        self.cache = []
        for ts, label in zip(ts_list, y):
            per_mod = _modality_features(ts, groups, order, source_mode)
            self.cache.append((per_mod, int(label)))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        per_mod, label = self.cache[idx]
        feat = [torch.from_numpy(f).float() for f in per_mod]
        noise_masks = tuple(torch.zeros_like(t) for t in feat)
        batch_data = tuple(feat) + (torch.tensor(label, dtype=torch.long),)
        sample_corr = ["none"] * len(feat)
        corr_modalities = torch.zeros(len(feat), dtype=torch.bool)
        return batch_data, (idx, sample_corr, corr_modalities), noise_masks


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _split_indices(y, subjects, sites, cv, n_splits, fold, seed):
    """Return (train_idx, val_idx, test_idx). val = held-out slice of train (early stop)."""
    idx = np.arange(len(y))
    if cv == "site":
        usites = sorted(set(sites.tolist()))
        test_site = usites[fold % len(usites)]
        test_idx = idx[sites == test_site]
        rest = idx[sites != test_site]
    elif cv == "subject":
        subs = np.unique(subjects)
        gkf = GroupKFold(n_splits=n_splits)
        # deterministic reshuffle by seed so repeats differ
        rng = np.random.default_rng(seed)
        pos = {s: i for i, s in enumerate(rng.permutation(subs))}
        sh = np.array([pos[s] for s in subjects])
        splits = list(gkf.split(idx, y, sh))
        train_full, test_idx = splits[fold % n_splits]
        rest = train_full
    else:
        raise ValueError(cv)
    # carve a val set out of `rest` by subject (early-stopping), ~15%
    rest_subs = np.unique(subjects[rest])
    rng = np.random.default_rng(seed + 1)
    rng.shuffle(rest_subs)
    n_val = max(1, int(0.15 * len(rest_subs)))
    val_subs = set(rest_subs[:n_val].tolist())
    val_idx = np.array([i for i in rest if subjects[i] in val_subs])
    train_idx = np.array([i for i in rest if subjects[i] not in val_subs])
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_modality_dims(data_dir: str, modality_grouping: str = "yeo7",
                      source_mode: str = "rowblock", **kwargs):
    """(name, in_dim) per modality, for config introspection. Loads data to get n_roi."""
    ts_list, *_ = _load_abide(data_dir)
    n_roi = ts_list[0].shape[1]
    groups = _yeo7_groups(n_roi)
    order = list(groups)
    out = []
    for g in order:
        k = len(groups[g])
        in_dim = k * n_roi if source_mode != "within" else k * (k - 1) // 2
        out.append((g, in_dim))
    return out


def get_dataloader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    noise_severity=None,        # API parity; unused
    test_noise: float = 0.0,    # API parity; unused
    exp_setup: str = "",        # API parity; unused
    modality_grouping: str = "yeo7",
    source_mode: str = "rowblock",
    cv: str = "subject",
    n_splits: int = 5,
    fold: int = 0,
    seed: int = 1000,
    **kwargs,
):
    """Return (train_loader, val_loader, test_loader) for ABIDE Yeo-7 C²MF."""
    ts_list, y, subjects, sites = _load_abide(data_dir)
    n_roi = ts_list[0].shape[1]
    groups = _yeo7_groups(n_roi)
    order = list(groups)
    dims = [(g, (len(groups[g]) * n_roi if source_mode != "within"
                 else len(groups[g]) * (len(groups[g]) - 1) // 2)) for g in order]
    print(f"[abide] grouping={modality_grouping} source_mode={source_mode} "
          f"M={len(order)} cv={cv} fold={fold} seed={seed}", flush=True)
    for g, d in dims:
        print(f"[abide]   {g:>12s}: k={len(groups[g]):>3d}  in_dim={d}", flush=True)

    tr, va, te = _split_indices(y, subjects, sites, cv, n_splits, fold, seed)
    print(f"[abide] split: train={len(tr)} val={len(va)} test={len(te)} "
          f"(test {'site=' + sorted(set(sites))[fold % len(set(sites))] if cv == 'site' else 'fold ' + str(fold)})",
          flush=True)

    def _ds(sel):
        return AbideFCDataset([ts_list[i] for i in sel], y[sel], groups, order, source_mode)

    train_loader = DataLoader(_ds(tr), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(_ds(va), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(_ds(te), batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/tmp/abide")
    ap.add_argument("--source_mode", default="rowblock")
    ap.add_argument("--cv", default="subject")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    tl, vl, te = get_dataloader(args.data_dir, source_mode=args.source_mode, cv=args.cv, fold=args.fold)
    xb = next(iter(tl))
    bd = xb[0]
    print("n_modalities+label:", len(bd), "mod0 shape:", bd[0].shape, "label sample:", bd[-1][:5])
