from __future__ import annotations

from typing import Union

import numpy as np
import torch
from pyts.approximation import (
    PiecewiseAggregateApproximation,
    SymbolicAggregateApproximation,
)
from tabpfn import TabPFNClassifier


def make_encoder(
    in_dim,
    embed_dim,
    n_layers,
    n_hidden,
    activation='torch.nn.Tanh()',
    dropout=0.0
):
    layers = [
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=in_dim, out_features=n_hidden),
        eval(activation),
    ]

    for _ in range(n_layers):
        if dropout > 0.0:
            layers += [
                torch.nn.Linear(n_hidden, n_hidden),
                eval(activation),
                torch.nn.Dropout(dropout),
            ]
        else:
            layers += [
                torch.nn.Linear(n_hidden, n_hidden),
                eval(activation),
            ]

    layers += [
        torch.nn.Linear(n_hidden, embed_dim),
        eval(activation),
    ]

    return torch.nn.Sequential(*layers)


def make_head(
    embed_dim,
    out_dim,
    final_activation='torch.nn.Softmax(dim=-1)'
):
    layers = [torch.nn.Linear(embed_dim, out_dim)]

    if eval(final_activation) is not None:
        layers += [eval(final_activation)]

    return torch.nn.Sequential(*layers)


class MLPEncoder(torch.nn.Module):
    """Plain feed-forward encoder for tabular / pre-extracted features.

    Used by the Frankfurt fMRI pipeline (FC upper-triangular vectors per
    modality).  Accepts `freeze_params` to match the FusionModel kwargs
    convention; everything else is the same as `make_encoder`.
    """

    def __init__(
        self,
        in_dim,
        embed_dim=64,
        n_layers=2,
        n_hidden=128,
        activation='torch.nn.Tanh()',
        dropout=0.0,
        freeze_params=False,
    ):
        super().__init__()
        self.encoder = make_encoder(
            in_dim=in_dim,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            activation=activation,
            dropout=dropout,
        )
        if freeze_params:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x, **kwargs):
        return self.encoder(x)


class Classifier(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        n_layers,
        n_hidden,
        embed_dim=64,
        activation='torch.nn.Tanh()',
        final_activation='torch.nn.Softmax(dim=-1)',
        dropout=0.0
    ):
        super().__init__()

        self.encoder = make_encoder(
            in_dim=in_dim,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            activation=activation,
            dropout=dropout
        )

        self.head = make_head(
            embed_dim=embed_dim,
            out_dim=out_dim,
            final_activation=final_activation
        )

    def forward(self, x, context=None, return_embedding=False, **kwargs):
        x = torch.cat(x, dim=-1) if isinstance(x, list) else x

        z = self.encoder(x)          # [B, embed_dim]
        logits = self.head(z)        # [B, out_dim]

        if return_embedding:
            return logits, z
        return logits


# ---------------------------------------------------------------------------
# Foundation-model encoder: TabPFN + SAX
# ---------------------------------------------------------------------------
#
# Variant iii (per the F1 design): per-modality unimodal prediction comes from
# frozen TabPFN.predict_proba(SAX(x)), the learnable embedding h_i is
# Linear(SAX(x)). TabPFN is not a torch.nn.Module and stays out of state_dict.
#
# Per-outer-fold lifecycle:
#   1. fit_tabpfn(X_train, y_train)  -- store in-context demonstrations
#   2. precompute_probs(X_all)       -- content-hash cache for forward path
#   3. forward(x) / predict_proba(x) -- hash-cache lookup, KeyError on miss
#
class TabPFNSAXEncoder(torch.nn.Module):
    """SAX + TabPFN encoder for ROI time-series modalities."""

    def __init__(
        self,
        roi_indices,
        embed_dim: int = 64,
        sax_alphabet: int = 8,
        sax_word_size: int = 16,
        sax_strategy: str = "quantile",
        n_classes: int = 3,
        tabpfn_device: str = "cpu",
        freeze_params: bool = False,
        random_state: int = 42,
        **_ignored,
    ):
        super().__init__()
        self.roi_indices = list(roi_indices)
        self.embed_dim = int(embed_dim)
        self.sax_alphabet = int(sax_alphabet)
        self.sax_word_size = int(sax_word_size)
        self.sax_strategy = str(sax_strategy)
        self.n_classes = int(n_classes)
        self.tabpfn_device = str(tabpfn_device)
        self.random_state = int(random_state)

        self.feat_dim = len(self.roi_indices) * self.sax_word_size
        self.projector = torch.nn.Linear(self.feat_dim, self.embed_dim)
        if freeze_params:
            for p in self.projector.parameters():
                p.requires_grad = False

        self._char_to_int = {chr(ord("a") + i): i for i in range(self.sax_alphabet)}

        # TabPFNClassifier intentionally NOT a submodule (must stay frozen and
        # outside autograd + state_dict).
        self._tabpfn = None
        # Content-hash caches populated by precompute_probs(); keyed by
        # hash(ts_slice.tobytes()) so train/val/test never collide on
        # dataset-local indices.
        self._probs_cache: dict[int, np.ndarray] = {}
        self._sax_cache: dict[int, np.ndarray] = {}

    def _sax_encode_batch(self, ts_batch: np.ndarray) -> np.ndarray:
        """SAX-encode (B, T, k_rois) -> (B, k_rois * sax_word_size) int8."""
        if ts_batch.ndim != 3:
            raise ValueError(f"expects (B, T, k_rois); got {ts_batch.shape}")
        _, _, k = ts_batch.shape
        if k != len(self.roi_indices):
            raise ValueError(
                f"k_rois mismatch: encoder expects {len(self.roi_indices)}, got {k}"
            )
        paa = PiecewiseAggregateApproximation(
            window_size=None, output_size=self.sax_word_size
        )
        sax = SymbolicAggregateApproximation(
            n_bins=self.sax_alphabet, strategy=self.sax_strategy
        )
        feats_per_roi = []
        for roi_local_idx in range(k):
            ts_roi = ts_batch[:, :, roi_local_idx]
            ts_paa = paa.fit_transform(ts_roi)
            ts_sax = sax.fit_transform(ts_paa)
            ids = np.array(
                [[self._char_to_int[c] for c in row] for row in ts_sax],
                dtype=np.int8,
            )
            feats_per_roi.append(ids)
        return np.concatenate(feats_per_roi, axis=1)

    def _slice_to_modality(self, ts: np.ndarray) -> np.ndarray:
        """Return ts as (n, T, k_rois). Accepts pre-sliced or full-132 atlas."""
        if ts.shape[-1] == len(self.roi_indices):
            return ts
        if ts.shape[-1] >= max(self.roi_indices) + 1:
            return ts[:, :, self.roi_indices]
        raise ValueError(
            f"Expected last-dim {len(self.roi_indices)} or full atlas "
            f"(>= {max(self.roi_indices) + 1}); got {ts.shape[-1]}"
        )

    def fit_tabpfn(self, X_train_full_ts: np.ndarray, y_train: np.ndarray) -> None:
        """Fit in-context TabPFN on the modality's SAX features."""
        if X_train_full_ts.ndim != 3:
            raise ValueError(
                f"X_train_full_ts must be (n, T, n_rois); got {X_train_full_ts.shape}"
            )
        ts_mod = self._slice_to_modality(X_train_full_ts)
        feats = self._sax_encode_batch(ts_mod).astype(np.float32)
        self._tabpfn = TabPFNClassifier(
            device=self.tabpfn_device, random_state=self.random_state
        )
        self._tabpfn.fit(feats, np.asarray(y_train).astype(np.int64))
        self._probs_cache = {}
        self._sax_cache = {}

    def precompute_probs(self, X_all_full_ts: np.ndarray, sample_indices=None) -> None:
        """Populate hash-keyed probs + SAX caches for the full fold.

        `sample_indices` is accepted for API parity with older callers but
        unused; cache is content-hash-keyed so train/val/test never collide.
        """
        del sample_indices  # unused; retained for caller compatibility
        if self._tabpfn is None:
            raise RuntimeError("precompute_probs requires fit_tabpfn first.")
        ts_mod = self._slice_to_modality(X_all_full_ts)
        feats = self._sax_encode_batch(ts_mod).astype(np.float32)
        with torch.no_grad():
            probs = self._tabpfn.predict_proba(feats)
        for row in range(ts_mod.shape[0]):
            h = int(hash(ts_mod[row].tobytes()))
            self._probs_cache[h] = probs[row].astype(np.float32)
            self._sax_cache[h] = feats[row]

    def _hash_lookup(self, x_np: np.ndarray, cache: dict) -> np.ndarray:
        """Stacked cache rows for x_np; raise KeyError on any miss."""
        rows = []
        for i in range(x_np.shape[0]):
            h = int(hash(x_np[i].tobytes()))
            if h not in cache:
                raise KeyError(
                    "TabPFNSAXEncoder cache miss; call precompute_probs() "
                    "with this fold's full sample set before forward/predict_proba."
                )
            rows.append(cache[h])
        return np.stack(rows, axis=0).astype(np.float32)

    def predict_proba(
        self,
        time_series_batch: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """TabPFN class probabilities for the batch (hash-cache lookup, B x n_classes)."""
        if self._tabpfn is None:
            raise RuntimeError("predict_proba called before fit_tabpfn.")
        if isinstance(time_series_batch, torch.Tensor):
            ts_np = time_series_batch.detach().cpu().numpy()
        else:
            ts_np = np.asarray(time_series_batch)
        ts_np = self._slice_to_modality(ts_np)
        probs_np = self._hash_lookup(ts_np, self._probs_cache)
        return torch.from_numpy(probs_np)

    def forward(
        self,
        x: Union[torch.Tensor, np.ndarray],
        **kwargs,
    ) -> torch.Tensor:
        """SAX-encode x, project to embed_dim. Hash-cache lookup when warm,
        live SAX-compute fallback when cold (empty cache or partial miss).
        The fallback is what lets FusionModel.noise_encoders work: those are
        deepcopies of self.encoders taken in __init__ (before fit_tabpfn),
        so their _sax_cache is always empty.
        """
        if isinstance(x, np.ndarray):
            x_np = x
        else:
            x_np = x.detach().cpu().numpy()
        if x_np.ndim != 3:
            raise ValueError(f"expects (B, T, k_rois); got {x_np.shape}")
        x_np = self._slice_to_modality(x_np)
        try:
            feats = self._hash_lookup(x_np, self._sax_cache)
        except KeyError:
            feats = self._sax_encode_batch(x_np).astype(np.float32)
        feats_t = torch.from_numpy(feats).to(
            next(self.projector.parameters()).device
        )
        return self.projector(feats_t)


# ---------------------------------------------------------------------------
# F2.1: TabPFN(FC) for p_i, Linear(SAX) for h_i unchanged.
# ---------------------------------------------------------------------------
#
# Single-knob test (vs Sanity-Sweep TabPFN baseline): switch ONLY TabPFN's
# input from SAX-tokens to FC-features (Pearson upper-triangle).  Orthogonal
# to F2.0 (which changed h_i).  Motivated by F2.0 outcome: h_i is NOT the
# bottleneck on dopM3, so the prime suspect is p_i, i.e. TabPFN's input format.
#
class TabPFNFCEncoder(TabPFNSAXEncoder):
    """TabPFN over FC features for p_i. h_i pipeline (Linear(SAX)) inherited."""

    def __init__(
        self,
        roi_indices,
        embed_dim: int = 64,
        sax_alphabet: int = 4,
        sax_word_size: int = 8,
        sax_strategy: str = "quantile",
        n_classes: int = 3,
        tabpfn_device: str = "cpu",
        freeze_params: bool = False,
        random_state: int = 42,
        **_ignored,
    ):
        super().__init__(
            roi_indices=roi_indices,
            embed_dim=embed_dim,
            sax_alphabet=sax_alphabet,
            sax_word_size=sax_word_size,
            sax_strategy=sax_strategy,
            n_classes=n_classes,
            tabpfn_device=tabpfn_device,
            freeze_params=freeze_params,
            random_state=random_state,
        )
        k = len(self.roi_indices)
        if k < 2:
            raise ValueError(f"TabPFNFCEncoder needs k>=2 ROIs for FC; got {k}")
        self.fc_dim = k * (k - 1) // 2

    def _compute_fc(self, ts_batch_np: np.ndarray) -> np.ndarray:
        """(B, T, k_rois) -> (B, k*(k-1)/2) Pearson upper-triangle float32."""
        B, _, k = ts_batch_np.shape
        if k != len(self.roi_indices):
            raise ValueError(
                f"k_rois mismatch: expected {len(self.roi_indices)}, got {k}"
            )
        triu_i, triu_j = np.triu_indices(k, k=1)
        out = np.empty((B, triu_i.size), dtype=np.float32)
        for b in range(B):
            with np.errstate(invalid='ignore', divide='ignore'):
                corr = np.corrcoef(ts_batch_np[b].T)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            out[b] = corr[triu_i, triu_j].astype(np.float32)
        return out

    def fit_tabpfn(self, X_train_full_ts: np.ndarray, y_train: np.ndarray) -> None:
        """Override: fit TabPFN's in-context demos on FC features (not SAX)."""
        if X_train_full_ts.ndim != 3:
            raise ValueError(
                f"X_train_full_ts must be (n, T, n_rois); got {X_train_full_ts.shape}"
            )
        ts_mod = self._slice_to_modality(X_train_full_ts)
        feats = self._compute_fc(ts_mod).astype(np.float32)
        self._tabpfn = TabPFNClassifier(
            device=self.tabpfn_device, random_state=self.random_state
        )
        self._tabpfn.fit(feats, np.asarray(y_train).astype(np.int64))
        self._probs_cache = {}
        self._sax_cache = {}

    def precompute_probs(self, X_all_full_ts: np.ndarray, sample_indices=None) -> None:
        """Override: predict_proba on FC; also populate SAX cache for h_i path."""
        del sample_indices
        if self._tabpfn is None:
            raise RuntimeError("precompute_probs requires fit_tabpfn first.")
        ts_mod = self._slice_to_modality(X_all_full_ts)
        fc_feats = self._compute_fc(ts_mod).astype(np.float32)
        with torch.no_grad():
            probs = self._tabpfn.predict_proba(fc_feats)
        # h_i forward path stays SAX-based — populate sax cache here too.
        sax_feats = self._sax_encode_batch(ts_mod).astype(np.float32)
        for row in range(ts_mod.shape[0]):
            h = int(hash(ts_mod[row].tobytes()))
            self._probs_cache[h] = probs[row].astype(np.float32)
            self._sax_cache[h] = sax_feats[row]
