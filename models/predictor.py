from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch


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
# F1 Foundation-Model encoder: TabPFN + SAX
# ---------------------------------------------------------------------------
#
# Drop-in replacement for `MLPEncoder` (same kwargs surface where it overlaps,
# same `forward(x, **kwargs)` shape) but the inputs are full ROI time series
# (B, T, k_rois) instead of FC-feature vectors, and TabPFN provides
# per-modality class probabilities `p_i` that are wired into the C2MF
# hyper-network as the credibility signal.
#
# Two operative subtleties (see vault/results/2026-05-14_F1.1_encoder_class_pre.md):
#   1. TabPFN is not differentiable — its frozen pretrained weights stay frozen.
#      The TabPFN forward is wrapped in `torch.no_grad()` and outputs are
#      `.detach()`'d. Backprop only runs through `self.projector` and the
#      downstream C²MF layers.
#   2. TabPFN needs `.fit(X_train, y_train)` to memorize demonstrations before
#      inference. This is called once per outer fold from the Lightning
#      LightningModule's `setup('fit')` hook via `encoder.fit_tabpfn(...)`.
#      An optional per-fold `precompute_probs(...)` cache makes the forward
#      pass a lookup instead of a TabPFN re-inference.
#
# Architectural choice (variant iii):
#   p_i := TabPFNClassifier.predict_proba(SAX(x))       # frozen, no grad
#   h_i := Linear(SAX(x))                                # learnable
#
class TabPFNSAXEncoder(torch.nn.Module):
    """SAX + TabPFN encoder for ROI time-series modalities.

    Drop-in replacement for `MLPEncoder` in the C²MF Frankfurt pipeline.

    Parameters
    ----------
    roi_indices : list[int]
        0-131 indices of the modality's ROIs (the columns of the full 132-ROI
        time-series tensor that belong to this modality bucket).
    embed_dim : int, default 64
        Output dimension of the embedding `h_i`.
    sax_alphabet : int, default 8
        Number of SAX quantile bins per ROI.
    sax_word_size : int, default 16
        Number of PAA segments per ROI (length of the SAX word).
    sax_strategy : str, default "quantile"
        SAX bucketization strategy (passed to
        `pyts.approximation.SymbolicAggregateApproximation`).
    n_classes : int, default 3
        Number of TabPFN output classes (Frankfurt: Amisulpride/LDopa/Placebo).
    tabpfn_device : str, default "cpu"
        Device for the TabPFNClassifier internal forward.
    freeze_params : bool, default False
        If True, freezes the learnable projector. TabPFN itself is always
        frozen regardless of this flag.
    random_state : int, default 42
        Passed to TabPFNClassifier for reproducibility.
    """

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
        # Lazy import: keep the module-level import surface clean so that
        # missing tabpfn / pyts in some envs (e.g. legacy CI) does not break
        # the rest of `predictor.py` consumers.
        try:
            from pyts.approximation import (  # noqa: F401
                PiecewiseAggregateApproximation,
                SymbolicAggregateApproximation,
            )
            from tabpfn import TabPFNClassifier  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "TabPFNSAXEncoder requires `tabpfn` and `pyts`. "
                "Install via: pip install tabpfn==2.2.1 pyts==0.13.0"
            ) from e

        self.roi_indices = list(roi_indices)
        self.embed_dim = int(embed_dim)
        self.sax_alphabet = int(sax_alphabet)
        self.sax_word_size = int(sax_word_size)
        self.sax_strategy = str(sax_strategy)
        self.n_classes = int(n_classes)
        self.tabpfn_device = str(tabpfn_device)
        self.random_state = int(random_state)

        # Learnable projector: concatenated SAX tokens (k_rois * word_size, as
        # float-cast integer ids) -> embed_dim.  This is the only trainable
        # piece of the encoder; TabPFN stays frozen.
        self.feat_dim = len(self.roi_indices) * self.sax_word_size
        self.projector = torch.nn.Linear(self.feat_dim, self.embed_dim)

        if freeze_params:
            for p in self.projector.parameters():
                p.requires_grad = False

        # Char -> int lookup for SAX output. SAX returns chars 'a','b',...
        self._char_to_int = {
            chr(ord("a") + i): i for i in range(self.sax_alphabet)
        }

        # Lazy TabPFN + per-fold cache, populated by `fit_tabpfn` /
        # `precompute_probs`.  We intentionally do NOT register the
        # TabPFNClassifier as a submodule — it must stay outside the
        # autograd graph and outside `self.parameters()`.
        self._tabpfn = None
        self._probs_cache: dict[int, np.ndarray] = {}

    # -----------------------------------------------------------------
    # SAX encoding helper
    # -----------------------------------------------------------------
    def _sax_encode_batch(self, ts_batch: np.ndarray) -> np.ndarray:
        """SAX-encode a batch of multi-ROI time series.

        Parameters
        ----------
        ts_batch : np.ndarray of shape (B, T, k_rois)
            Raw ROI time series for the modality.

        Returns
        -------
        feats : np.ndarray of shape (B, k_rois * sax_word_size), dtype int8
            Concatenated SAX token-id features (one row per sample).
        """
        from pyts.approximation import (
            PiecewiseAggregateApproximation,
            SymbolicAggregateApproximation,
        )

        if ts_batch.ndim != 3:
            raise ValueError(
                f"_sax_encode_batch expects (B, T, k_rois); got {ts_batch.shape}"
            )
        B, T, k = ts_batch.shape
        if k != len(self.roi_indices):
            raise ValueError(
                f"k_rois mismatch: encoder expects {len(self.roi_indices)} "
                f"ROIs, got tensor with {k}"
            )

        paa = PiecewiseAggregateApproximation(
            window_size=None, output_size=self.sax_word_size
        )
        sax = SymbolicAggregateApproximation(
            n_bins=self.sax_alphabet, strategy=self.sax_strategy
        )

        feats_per_roi = []
        for roi_local_idx in range(k):
            ts_roi = ts_batch[:, :, roi_local_idx]  # (B, T)
            ts_paa = paa.fit_transform(ts_roi)
            ts_sax = sax.fit_transform(ts_paa)
            ids = np.array(
                [[self._char_to_int[c] for c in row] for row in ts_sax],
                dtype=np.int8,
            )
            feats_per_roi.append(ids)
        return np.concatenate(feats_per_roi, axis=1)  # (B, k * word_size)

    # -----------------------------------------------------------------
    # TabPFN lifecycle (called from Lightning setup-hook, per fold)
    # -----------------------------------------------------------------
    def fit_tabpfn(
        self,
        X_train_full_ts: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """Fit the in-context TabPFN classifier on the modality SAX features.

        Slices the modality ROIs out of the full 132-ROI training tensor,
        SAX-encodes them, and stores demonstrations in
        `self._tabpfn`.  Called once per outer CV fold from the
        LightningModule's `setup('fit')` hook.

        Parameters
        ----------
        X_train_full_ts : np.ndarray of shape (n_train, T, 132)
            Full ROI time series for the training subjects.
        y_train : np.ndarray of shape (n_train,)
            Class labels.
        """
        from tabpfn import TabPFNClassifier

        if X_train_full_ts.ndim != 3:
            raise ValueError(
                f"X_train_full_ts must be (n, T, n_total_rois); got "
                f"{X_train_full_ts.shape}"
            )
        ts_mod = X_train_full_ts[:, :, self.roi_indices]
        feats = self._sax_encode_batch(ts_mod).astype(np.float32)

        self._tabpfn = TabPFNClassifier(
            device=self.tabpfn_device,
            random_state=self.random_state,
        )
        self._tabpfn.fit(feats, np.asarray(y_train).astype(np.int64))
        # Clear stale per-fold cache from previous fits.
        self._probs_cache = {}

    def precompute_probs(
        self,
        X_all_full_ts: np.ndarray,
        sample_indices,
    ) -> None:
        """Precompute and cache TabPFN probabilities for all samples in the fold.

        Optional optimisation: lets `forward` be a dict-lookup instead of a
        live TabPFN inference per batch.  Call from Lightning's
        `setup('fit')` hook right after `fit_tabpfn`.

        Parameters
        ----------
        X_all_full_ts : np.ndarray of shape (n_total, T, 132)
            All time series in the fold (train + val + test). Index alignment
            with `sample_indices` is the caller's responsibility.
        sample_indices : iterable[int]
            Global sample ids that map row positions of `X_all_full_ts` to
            integer keys used by `forward(sample_indices=...)`.
        """
        if self._tabpfn is None:
            raise RuntimeError(
                "precompute_probs requires fit_tabpfn first."
            )
        ts_mod = X_all_full_ts[:, :, self.roi_indices]
        feats = self._sax_encode_batch(ts_mod).astype(np.float32)
        with torch.no_grad():
            probs = self._tabpfn.predict_proba(feats)
        self._probs_cache = {
            int(idx): probs[row].astype(np.float32)
            for row, idx in enumerate(sample_indices)
        }

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------
    def predict_proba(
        self,
        time_series_batch: Optional[Union[torch.Tensor, np.ndarray]] = None,
        sample_indices=None,
    ) -> torch.Tensor:
        """TabPFN class probabilities for a batch of samples.

        Two paths:
          (a) If `sample_indices` are given AND the cache is populated, lookup.
          (b) Else: live TabPFN inference on `time_series_batch`.

        Returns
        -------
        probs : torch.Tensor of shape (B, n_classes)
            Detached, no-grad, float32.
        """
        if self._tabpfn is None:
            raise RuntimeError(
                "predict_proba called before fit_tabpfn — TabPFN has no "
                "in-context demonstrations yet."
            )

        if sample_indices is not None and len(self._probs_cache) > 0:
            rows = []
            for idx in sample_indices:
                key = int(idx.item() if hasattr(idx, "item") else idx)
                if key not in self._probs_cache:
                    raise KeyError(
                        f"sample_index {key} not in TabPFN probs cache; call "
                        "precompute_probs with this fold's sample ids."
                    )
                rows.append(self._probs_cache[key])
            probs_np = np.stack(rows, axis=0).astype(np.float32)
            return torch.from_numpy(probs_np)

        # Live inference fallback (slow; only for debug / sanity).
        if time_series_batch is None:
            raise ValueError(
                "predict_proba needs either sample_indices (with cache) or "
                "a time_series_batch for live inference."
            )
        if isinstance(time_series_batch, torch.Tensor):
            ts_np = time_series_batch.detach().cpu().numpy()
        else:
            ts_np = np.asarray(time_series_batch)
        # Caller may pass either (B, T, 132) full or (B, T, k_rois) sliced.
        if ts_np.shape[-1] != len(self.roi_indices):
            ts_np = ts_np[:, :, self.roi_indices]
        feats = self._sax_encode_batch(ts_np).astype(np.float32)
        with torch.no_grad():
            probs = self._tabpfn.predict_proba(feats)
        return torch.from_numpy(probs.astype(np.float32))

    # -----------------------------------------------------------------
    # PyTorch forward (learnable path)
    # -----------------------------------------------------------------
    def forward(
        self,
        x: Union[torch.Tensor, np.ndarray],
        sample_indices=None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass: SAX-encode `x`, project to embed_dim.

        Parameters
        ----------
        x : torch.Tensor or np.ndarray of shape (B, T, k_rois) OR (B, T, 132)
            ROI time series for the batch.  If the last dim equals 132, the
            modality slice is taken internally; if it equals `len(roi_indices)`,
            it is treated as already-sliced.
        sample_indices : optional
            Unused inside `forward` but kept in the signature so that
            `predict_proba(sample_indices=...)` can be called separately on
            the same batch by the C²MF wrapper.

        Returns
        -------
        h : torch.Tensor of shape (B, embed_dim)
            Learnable modality embedding.  Differentiable w.r.t. the projector.
        """
        # Accept torch or numpy; preserve dtype for the projector.
        if isinstance(x, np.ndarray):
            x_np = x
        else:
            x_np = x.detach().cpu().numpy()

        if x_np.ndim != 3:
            raise ValueError(
                f"TabPFNSAXEncoder.forward expects (B, T, k_rois); got "
                f"{x_np.shape}"
            )
        if x_np.shape[-1] != len(self.roi_indices):
            if x_np.shape[-1] >= max(self.roi_indices) + 1:
                x_np = x_np[:, :, self.roi_indices]
            else:
                raise ValueError(
                    f"Expected last-dim {len(self.roi_indices)} or full atlas; "
                    f"got {x_np.shape[-1]}"
                )

        feats = self._sax_encode_batch(x_np).astype(np.float32)  # (B, feat_dim)
        feats_t = torch.from_numpy(feats).to(
            next(self.projector.parameters()).device
        )
        h = self.projector(feats_t)  # (B, embed_dim)
        return h
