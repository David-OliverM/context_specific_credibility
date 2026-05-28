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
        """Fit in-context TabPFN on the modality's SAX features.

        ``ignore_pretraining_limits=True`` carried forward from F2.2 — no-op
        for vocabs that stay under TabPFN's 500-feature limit, required for
        higher word_size variants.
        """
        if X_train_full_ts.ndim != 3:
            raise ValueError(
                f"X_train_full_ts must be (n, T, n_rois); got {X_train_full_ts.shape}"
            )
        ts_mod = self._slice_to_modality(X_train_full_ts)
        feats = self._sax_encode_batch(ts_mod).astype(np.float32)
        self._tabpfn = TabPFNClassifier(
            device=self.tabpfn_device,
            random_state=self.random_state,
            ignore_pretraining_limits=True,
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
# F2.3: TabPFN-internal embedding as h_i (instead of Linear(SAX))
# ---------------------------------------------------------------------------
#
# Single-knob test vs Sanity baseline: h_i comes from TabPFN's transformer-
# internal representation (clf.get_embeddings, averaged across estimators)
# projected to embed_dim by a small learnable Linear.  p_i = TabPFN.predict_proba
# unchanged.  Both use the same SAX-input features per Sanity-Sweep vocab
# (a=4, w=8), unless overridden via Hydra.
#
# Embedding API (tabpfn 2.2.1):
#   clf.get_embeddings(X, data_source="test") -> (n_estimators=8, n_samples, 192)
#   We mean over estimators -> (n_samples, 192) -> learnable Linear -> (n_samples, embed_dim).
#
class TabPFNEmbeddingEncoder(TabPFNSAXEncoder):
    """TabPFN-internal embedding as h_i.  F2.3."""

    TABPFN_EMBED_DIM = 192  # tabpfn 2.x model dim; verified empirically

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
            freeze_params=False,
            random_state=random_state,
        )
        # Replace inherited SAX-linear projector with a TabPFN-embed projector.
        del self.projector
        self.projector = torch.nn.Linear(self.TABPFN_EMBED_DIM, self.embed_dim)
        if freeze_params:
            for p in self.projector.parameters():
                p.requires_grad = False
        # New cache: hash(ts.tobytes()) -> mean-over-estimators embedding (192,)
        self._embed_cache: dict[int, np.ndarray] = {}

    def precompute_probs(self, X_all_full_ts: np.ndarray, sample_indices=None) -> None:
        """Override: populate _probs_cache (TabPFN.predict_proba) AND
        _embed_cache (TabPFN.get_embeddings, mean over estimators)."""
        del sample_indices
        if self._tabpfn is None:
            raise RuntimeError("precompute_probs requires fit_tabpfn first.")
        ts_mod = self._slice_to_modality(X_all_full_ts)
        sax_feats = self._sax_encode_batch(ts_mod).astype(np.float32)
        with torch.no_grad():
            probs = self._tabpfn.predict_proba(sax_feats)
            # get_embeddings returns (n_estimators, n_samples, embed_dim)
            embed_raw = self._tabpfn.get_embeddings(sax_feats, data_source="test")
            # mean over estimators -> (n_samples, embed_dim)
            if embed_raw.ndim == 3:
                embed_mean = embed_raw.mean(axis=0)
            else:
                embed_mean = embed_raw
            embed_mean = np.asarray(embed_mean, dtype=np.float32)
        if embed_mean.shape[1] != self.TABPFN_EMBED_DIM:
            raise RuntimeError(
                f"TabPFN embedding dim mismatch: expected {self.TABPFN_EMBED_DIM}, "
                f"got {embed_mean.shape[1]}.  Update TABPFN_EMBED_DIM class const."
            )
        for row in range(ts_mod.shape[0]):
            h = int(hash(ts_mod[row].tobytes()))
            self._probs_cache[h] = probs[row].astype(np.float32)
            self._embed_cache[h] = embed_mean[row]
            # also populate _sax_cache for noise-encoder fallback in inherited forward
            self._sax_cache[h] = sax_feats[row]

    def _embed_lookup(self, x_np: np.ndarray) -> np.ndarray:
        rows = []
        for i in range(x_np.shape[0]):
            h = int(hash(x_np[i].tobytes()))
            if h not in self._embed_cache:
                raise KeyError(
                    "TabPFNEmbeddingEncoder embed cache miss; call "
                    "precompute_probs() first."
                )
            rows.append(self._embed_cache[h])
        return np.stack(rows, axis=0).astype(np.float32)

    def forward(self, x, **kwargs):
        if isinstance(x, np.ndarray):
            x_np = x
        else:
            x_np = x.detach().cpu().numpy()
        if x_np.ndim != 3:
            raise ValueError(f"expects (B, T, k_rois); got {x_np.shape}")
        x_np = self._slice_to_modality(x_np)
        # Try the embedding cache.  If miss (noise-encoder deepcopy path),
        # fall back to live get_embeddings via the encoder's own _tabpfn.
        try:
            emb_np = self._embed_lookup(x_np)
        except KeyError:
            if self._tabpfn is None:
                raise RuntimeError(
                    "TabPFNEmbeddingEncoder.forward cache miss AND no fitted "
                    "TabPFN — likely a noise_encoders deepcopy before fit. "
                    "Caller must call fit_tabpfn first."
                )
            sax_feats = self._sax_encode_batch(x_np).astype(np.float32)
            with torch.no_grad():
                emb_raw = self._tabpfn.get_embeddings(sax_feats, data_source="test")
                emb_np = (emb_raw.mean(axis=0) if emb_raw.ndim == 3 else emb_raw).astype(np.float32)
        emb_t = torch.from_numpy(emb_np).to(
            next(self.projector.parameters()).device
        )
        return self.projector(emb_t)
