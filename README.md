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

    
