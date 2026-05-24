# Yeo-7 Mapping: v1 (heuristic) vs v2 (MNI overlay)

- Total ROIs: 132
- Agreements: 102  (cortical-only: 61 / 91)
- Disagreements: 30  (cortical-only: 30 / 91)
- Unmatched HO labels: 0

## Disagreements (cortical) + unmatched

| idx | token | long_name | v1 | v2 | n_voxels | top-3 voxel-counts (Yeo-label: n) |
|---:|:---|:---|:---|:---|---:|:---|
| 1 | FP_L | Frontal Pole | CONT | DEFAULT | 7061 | DEFAULT:2771, CONT:2059, LIMBIC:789 |
| 4 | SFG_R | Superior Frontal Gyrus | CONT | DEFAULT | 2412 | DEFAULT:837, CONT:626, VENTATTN:389 |
| 5 | SFG_L | Superior Frontal Gyrus | CONT | DEFAULT | 2538 | DEFAULT:1317, CONT:407, DORSATTN:363 |
| 8 | IFGtri_R | Inferior Frontal Gyrus, pars triangularis | CONT | DEFAULT | 496 | DEFAULT:236, CONT:215, VENTATTN:33 |
| 9 | IFGtri_L | Inferior Frontal Gyrus, pars triangularis | CONT | DEFAULT | 607 | DEFAULT:371, CONT:216, VENTATTN:1 |
| 10 | IFGoper_R | Inferior Frontal Gyrus, pars opercularis | VENTATTN | CONT | 655 | CONT:321, VENTATTN:160, DORSATTN:113 |
| 11 | IFGoper_L | Inferior Frontal Gyrus, pars opercularis | VENTATTN | DEFAULT | 729 | DEFAULT:321, CONT:237, VENTATTN:111 |
| 16 | aSTG_R | Superior Temporal Gyrus, anterior division | LIMBIC | DEFAULT | 268 | DEFAULT:146, SOMMOT:118 |
| 17 | aSTG_L | Superior Temporal Gyrus, anterior division | LIMBIC | DEFAULT | 253 | DEFAULT:167, SOMMOT:86 |
| 19 | pSTG_L | Superior Temporal Gyrus, posterior division | SOMMOT | DEFAULT | 877 | DEFAULT:517, SOMMOT:333 |
| 24 | toMTG_R | Middle Temporal Gyrus, temporooccipital part | DEFAULT | DORSATTN | 1151 | DORSATTN:424, VENTATTN:257, DEFAULT:202 |
| 25 | toMTG_L | Middle Temporal Gyrus, temporooccipital part | DEFAULT | CONT | 853 | CONT:222, DEFAULT:216, DORSATTN:209 |
| 26 | aITG_R | Inferior Temporal Gyrus, anterior division | DEFAULT | LIMBIC | 317 | LIMBIC:285, DEFAULT:22 |
| 27 | aITG_L | Inferior Temporal Gyrus, anterior division | DEFAULT | LIMBIC | 330 | LIMBIC:220, DEFAULT:94 |
| 28 | pITG_R | Inferior Temporal Gyrus, posterior division | VIS | LIMBIC | 947 | LIMBIC:544, CONT:182, DORSATTN:83 |
| 29 | pITG_L | Inferior Temporal Gyrus, posterior division | VIS | LIMBIC | 995 | LIMBIC:446, DEFAULT:226, CONT:140 |
| 30 | toITG_R | Inferior Temporal Gyrus, temporooccipital part | VIS | DORSATTN | 774 | DORSATTN:542, CONT:133, VIS:79 |
| 31 | toITG_L | Inferior Temporal Gyrus, temporooccipital part | VIS | DORSATTN | 696 | DORSATTN:431, CONT:239 |
| 38 | pSMG_R | Supramarginal Gyrus, posterior division | VENTATTN | CONT | 1207 | CONT:564, VENTATTN:389, DORSATTN:100 |
| 39 | pSMG_L | Supramarginal Gyrus, posterior division | VENTATTN | CONT | 1052 | CONT:430, VENTATTN:265, DEFAULT:245 |
| 48 | MedFC | Frontal Medial Cortex | DEFAULT | LIMBIC | 976 | LIMBIC:540, DEFAULT:369 |
| 52 | PaCiG_R | Paracingulate Gyrus | VENTATTN | DEFAULT | 1369 | DEFAULT:721, CONT:311, VENTATTN:258 |
| 53 | PaCiG_L | Paracingulate Gyrus | VENTATTN | DEFAULT | 1336 | DEFAULT:835, CONT:229, VENTATTN:216 |
| 60 | FOrb_L | Frontal Orbital Cortex | LIMBIC | DEFAULT | 1703 | DEFAULT:721, LIMBIC:671, CONT:75 |
| 63 | pPaHC_R | Parahippocampal Gyrus, posterior division | LIMBIC | VIS | 320 | VIS:211, DEFAULT:47, LIMBIC:29 |
| 64 | pPaHC_L | Parahippocampal Gyrus, posterior division | LIMBIC | DEFAULT | 390 | DEFAULT:190, VIS:136, LIMBIC:21 |
| 67 | aTFusC_R | Temporal Fusiform Cortex, anterior division | VIS | LIMBIC | 294 | LIMBIC:280 |
| 68 | aTFusC_L | Temporal Fusiform Cortex, anterior division | VIS | LIMBIC | 316 | LIMBIC:287 |
| 69 | pTFusC_R | Temporal Fusiform Cortex, posterior division | VIS | LIMBIC | 721 | LIMBIC:417, VIS:258, DORSATTN:7 |
| 70 | pTFusC_L | Temporal Fusiform Cortex, posterior division | VIS | LIMBIC | 873 | LIMBIC:433, VIS:249, DORSATTN:119 |

## Hemispheric symmetry enforcement (v2 -> v2_symmetric)

6 cortical ROIs were re-labeled when enforcing L=R via combined-voxel-argmax.

| idx | token | v2 raw | v2 symmetric |
|---:|:---|:---|:---|
| 1 | FP_L | DEFAULT | CONT |
| 11 | IFGoper_L | DEFAULT | CONT |
| 19 | pSTG_L | DEFAULT | SOMMOT |
| 25 | toMTG_L | CONT | DORSATTN |
| 60 | FOrb_L | DEFAULT | LIMBIC |
| 64 | pPaHC_L | DEFAULT | VIS |
