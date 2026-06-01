# Context Specific Credibility aware Multimodal Fusion
This repository contains the code for the project - **[Context-specific Credibility-aware Multimodal Fusion with Conditional Probabilistic Circuits](https://arxiv.org/abs/2603.26629)**. This is under active development.


## Setup
Create a new virtual environment and install the required packages given in `requirements.txt`.

**Submodule Dependencies**
This repository has dependencies with following three packages. They are organized in the `packages` directory.
- [MultiBench](https://github.com/pliang279/MultiBench)
- [RatSPN](https://github.com/braun-steven/spn-pytorch-experiments)
- [EinsumNet](https://github.com/braun-steven/simple-einet)

## Datasets
This repository currently supports [NYUD](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) and [AVMNIST](https://github.com/yedizhang/audiovisual-mnist) datasets (can be found [here](https://drive.google.com/drive/folders/1Ij6koHLRNbPDI9reNOUQ1rLOy4CWqYqG?usp=sharing)). You can also find pretrained unimodal predictors in the same folder.

## To Run
Specify the hyperparameter configurations for your experiment in the appropriate config file inside `conf/`. 
Use the following commands to run experiments. You can pass values as needed from the command line for the hyperparameters specified in the config file.

**Joint Training trains the pipeline end-to-end. To only train the fusion function decoupled from the pretrained unimodal predictors, set fully_decoupled_training=True**.

**noise_severity sets the noise on train data - set it to Null if you don't want to explicitly add noise**
```bash
python main.py dataset=nyud2 experiment=nyud2_cs_credibility_weighted group_tag=base seed=42 exp_setup=joint_trng noise_severity=1 test_noise=0.5
```
```bash
python credibility.py dataset=nyud2 experiment=nyud2_cs_credibility_weighted group_tag=base seed=42 exp_setup=joint_trng noise_severity=1 test_noise=0.5
```

## Currently Supported Late Fusion Methods
- [x] Weighted Mean
- [x] Noisy-or
- [x] MLP
- [x] TMC
- [x] EinsumNet with Dirichlet leaves (Direct-PC)
- [x] Conditional-SPN
- [x] Credibility Weighted Mean
- [x] Context-Specific Credibility Weighted Mean

---

## Fork notes (David-OliverM/context_specific_credibility)

> This section is specific to our research fork and is not part of the upstream
> README. `upstream` = `Pranuthi23/context_specific_credibility` (read-only);
> `origin` = this fork. The fork adds a Frankfurt pharmacological-fMRI dataloader
> and a real-clinical-data evaluation of C²MF.

### Branch → purpose map

**Convention:** the `f*/` (encoder/grouping experiments) and `repro*/` (Tenali
reproduction) branches are kept SEPARATE from `main` and are never auto-merged;
they are promoted into `main` only by hand, and only when an experiment produced
a real improvement worth keeping.

| Branch | Purpose |
|---|---|
| `main` | Integration branch. Frankfurt dataloader (`dataloader/frankfurt/`), modality groupings (`conf/grouping/`: Yeo-7 + dopamine v3), sanity configs. Kept aligned with upstream where practical. Stable infra (e.g. the literature-grounded dopamine v3 grouping) lands here. |
| `f2.0/mlp-h-pipeline` | F2.0: `TabPFNSAXMLPEncoder` — MLP over FC features for the context embedding `h_i`; + Hydra configs. |
| `f2.1/fc-for-tabpfn` | F2.1: `TabPFNFCEncoder` — TabPFN over within-modality FC features (the backbone we standardised on). |
| `f2.2/sax-vocab-sweep` | F2.2: SAX vocabulary sweep; `TabPFN.fit(..., ignore_pretraining_limits=True)`. |
| `f2.3/tabpfn-embedding` | F2.3: `TabPFNEmbeddingEncoder` (TabPFN embeddings as `h_i`). |
| `f2.5/h-pi-stacking` | F2.5: stacking `h_i` (MLP+FC) onto the best `p_i` (SAX-w32 or FC). |
| `f2.5a-mr/multi-repeat` | Live multi-repeat-CV Frankfurt branch: `subject_shuffle_seed` harness for honest CIs, groupings, configs. 7 commits ahead of `main`. |
| `repro/avmnist-nyud-tenali2026` | R1 reproduction sweep of Tenali 2026 Table I (AVMNIST + NYUD), 5-seed, smoke test. |
| `repro-clean/pranuthi-plus-bugfixes` | Minimal upstream + bugfixes — the honest **Path A** (no corruption oracle in the hyper-network context). |
| `repro-clean-test/path-b-pranuthi` | Path A + the corruption-aware context concat — **Path B** (feeds the synthetic-corruption oracle); used to show the published C²MF gain depends on it. |
| `main-pre-rebase-2026-05-11` | Historical snapshot of `main` before the 2026-05-11 rebase (Frankfurt dataloader skeleton). Kept for reference. |

### Frankfurt modality groupings (`conf/grouping/`)

- **Yeo-7** (`yeo7_frankfurt_v2_symmetric.csv`, M=9): functional-network grouping
  via MNI overlay. Primary grouping.
- **Dopamine circuit** (`dopamine_circuit_v3.csv`, M=3 effective in the paper
  pipeline): literature-grounded reward-circuit grouping
  [Haber & Knutson 2010; Di Martino 2008]. See `conf/grouping/README.md` for the
  full per-bucket citation audit. `dopamine_circuit_v2.csv` (expert-sketch,
  identical membership) is kept for provenance.

    
