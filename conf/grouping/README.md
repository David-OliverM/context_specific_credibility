# Modality Grouping Artefacts for Frankfurt fMRI (C²MF)

> Status: two production mappings ship in this directory, both loaded at runtime
> by `dataloader/frankfurt/get_data.py`.
> Last verified: 2026-05-13.

This directory holds the CSV-based modality-bucket definitions that turn the
132 Frankfurt ROIs into a small set of C²MF modalities. Two mappings are
production-ready and selectable via the Hydra `modality_grouping` switch:

- **Yeo-7 functional networks** (`modality_grouping: yeo7`)
- **Dopamine-circuit after Lorina Zapf** (`modality_grouping: dopamine`)

A third option, `modality_grouping: anatomical`, is a tiny 3-bucket
sanity-baseline that lives in code (no CSV needed).

## Files

### Yeo-7 mapping (primary, M=9 functional networks)

| File | Status | Purpose |
|---|---|---|
| `yeo7_frankfurt_v2_symmetric.csv` | **production** | MNI-overlay verified mapping with L=R symmetry enforced. Used by the dataloader. |
| `yeo7_frankfurt_v2.csv` | audit | Raw per-hemisphere overlay result. Six L/R pairs disagree; resolved in v2_symmetric. |
| `yeo7_frankfurt_v1.csv` | superseded | Initial heuristic derived from anatomical-label reasoning. Kept for diff transparency. |
| `yeo7_v1_vs_v2_diff.md` | audit | Per-ROI agreement / disagreement report between v1 and v2 plus the symmetry-enforcement step. |

Reproducing script: `paper/scripts/verify_yeo7_mapping.py`. Vault documentation:
`vault/methods/Yeo-7 Mapping for Frankfurt ROIs.md`.

### Dopamine-circuit mapping (secondary, M=4 effective)

| File | Status | Purpose |
|---|---|---|
| `dopamine_circuit_v2.csv` | **production** | Lorina Zapf's explicit cortico-striato-mesencephalic dopamine circuit definition (UKF, 2026-05-07). |

The mapping has five nominal buckets (`dorsal_striatum`, `ventral_striatum`,
`midbrain_proxy`, `da_projection_cortex`, `other`). Brain-Stem with k=1 falls
through the k>=2 filter, reducing effective M to 4. Vault documentation:
`vault/methods/Dopamine-Circuit Mapping for Frankfurt ROIs.md`. There is no
v1 CSV in the repo because the pre-Lorina v1 mapping was a hard-coded regex
inside `_dopamine_grouping`; v2 is the first persisted version.

### Doc

| File | Purpose |
|---|---|
| `README.md` | this file. |

## Production mapping (v2_symmetric) bucket counts

| Bucket | k (ROIs) | FC features (k(k-1)/2) | Source |
|---|---|---|---|
| `VIS` | 18 | 153 | Yeo 2011 cortical |
| `SOMMOT` | 18 | 153 | Yeo 2011 cortical |
| `DORSATTN` | 8 | 28 | Yeo 2011 cortical |
| `VENTATTN` | 7 | 21 | Yeo 2011 cortical |
| `LIMBIC` | 16 | 120 | Yeo 2011 cortical |
| `CONT` | 8 | 28 | Yeo 2011 cortical |
| `DEFAULT` | 16 | 120 | Yeo 2011 cortical |
| `SUBCORTICAL` | 15 | 105 | auxiliary, Yeo-7 not defined off-cortex |
| `CEREBELLAR` | 26 | 325 | auxiliary, Yeo-7 not defined off-cortex |
| **Total** | **132** | **1053** | |

All nine buckets are non-empty (each has `k>=2` so the FC upper triangle is non-degenerate).  Compared to v1 the distribution is noticeably more balanced: `DORSATTN` doubled from 4 to 8 ROIs, removing the v1 concern that this bucket was too small.

## How v2_symmetric was derived

[`paper/scripts/verify_yeo7_mapping.py`](../../../paper/scripts/verify_yeo7_mapping.py) executes the following pipeline:

1. **Atlases via nilearn 0.13.1.**
   - Yeo 2011 7-network `Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz` (`datasets.fetch_atlas_yeo_2011()`)
   - Harvard-Oxford cortical `cort-maxprob-thr25-2mm` (`datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')`)
2. **Resample.** Yeo (1mm, 256³) is resampled to the HO grid (91×109×91, 2mm) with nearest-neighbour interpolation (`image.resample_to_img`).
3. **Per-ROI mask.** For each of the 48 HO bilateral cortical labels, build a binary mask. For paired regions, split by sagittal `x` coordinate (x<0 = L, x>0 = R). Five midline ROIs (MedFC, SubCalC, AC, PC, Precuneous) are kept as a single mask.
4. **Modal Yeo label.** For each ROI mask, count Yeo voxel labels (1-7, ignoring Background=0) and take argmax.
5. **L=R symmetry.** For each L/R pair, combine voxel-counts across hemispheres and re-take argmax, applying the result to both hemispheres. Six ROIs flipped during this step (FP_L, IFGoper_L, pSTG_L, toMTG_L, FOrb_L, pPaHC_L) - their raw voxel-count margin was small enough that the L-side argmax differed from R; symmetry-enforcement resolves to the combined-side argmax.
6. **Subcortical + cerebellar pass-through.** ROIs 91-131 are routed directly into the SUBCORTICAL / CEREBELLAR buckets without overlay (Yeo-7 is not defined off-cortex).

Label-name matching between Frankfurt CONN labels and HO labels uses a normalization step (strip trailing parenthetical annotations, strip dash-enclosed annotations, lowercase) plus a single alias `Operculum -> Opercular` to handle a Frankfurt source-file inconsistency (CONN labels FO and PO as "Operculum" but CO as "Opercular"; HO consistently uses "Opercular").

## v1 vs v2 agreement summary

- 61 / 91 cortical ROIs got the same bucket from v1 (heuristic) and v2 (overlay)
- 30 cortical ROIs were re-labeled by the overlay. Biggest movements:
  - 7 ROIs moved into LIMBIC (mostly ventral temporal: aITG, pITG, aTFusC, pTFusC; plus MedFC)
  - 8 ROIs moved into DEFAULT (mostly Yeo's broader DMN reach: SFG, IFGtri, aSTG, PaCiG, FOrb_L)
  - 4 ROIs moved into DORSATTN (toMTG, toITG, sLOC was already DORSATTN in v1)
  - VENTATTN shrunk from 13 to 7 ROIs (overlay placed PaCiG, pSMG, IFGoper into DEFAULT or CONT)

See `yeo7_v1_vs_v2_diff.md` for the per-ROI agreement table.

## Known limitations of v2_symmetric

1. **Threshold sensitivity.** The HO cort-maxprob-thr25-2mm atlas thresholds probabilities at 25%. Switching to thr0 or thr50 would change ROI mask sizes and potentially flip ambiguous assignments. Defensible default; revisit if a published reference uses a different threshold.
2. **Liberal mask version of Yeo.** Yeo 2011 ships a "Liberal" and a "Conservative" mask. We use the Liberal version (more voxels labeled, fewer Background). For ROIs near the cortical edge this may slightly bias toward Yeo's edge-class.
3. **Cerebellar still lumped into one bucket.** Buckner et al. (2011, J Neurophysiol 106:2322-2345) extended Yeo-7 to the cerebellum at the lobule level. A v3 could split the 26 cerebellar ROIs into Yeo-aligned sub-buckets using the Buckner template (also available via nilearn). This was not done in v2.
4. **No quantitative confidence per ROI.** The `note` column reports voxel counts and top-3 buckets; a future refinement could expose a per-ROI margin (e.g. `(top1 - top2) / total`) to flag the most ambiguous assignments at training time.
5. **Resting-state-derived parcellation applied to drug-state data.** Yeo 2011 was derived from healthy resting-state subjects. Applying it to pharmacological scans (Amisulpride / LDopa / Placebo) assumes the cortical parcellation is drug-state invariant. This is the standard assumption in the field but should be acknowledged.

## Citations

- Yeo et al. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. *J Neurophysiol*, 106(3), 1125-1165.
- Buckner et al. (2011). The organization of the human cerebellum estimated by intrinsic functional connectivity. *J Neurophysiol*, 106(5), 2322-2345.
- Harvard-Oxford cortical atlas: bundled with FSL; original release based on Desikan et al. (2006).
- AAL cerebellar atlas: Tzourio-Mazoyer et al. (2002).
- nilearn: Abraham et al. (2014). Machine learning for neuroimaging with scikit-learn. *Front Neuroinform*, 8:14.
