# Cross-Modal Benchmarking

In this folder we provide code for training the baseline cross-modal place recognition approach *LIP-Loc: LiDAR Image Pretraining for Cross-Modal Localization*. 

## Setting up the environment
We use the package management tool **mamba** for constructing python virtual environments to run our experiments in.  We provide an `environment.yaml` file which can be used to construct the virtual environment for these experiments using the following command:
```
mamba env create -f environment.yaml
```
## Config files
Config files for different network architectures (*e.g.* ResNet50, DinoV2, DinoV3) can be found in the `LIP-Loc/configs` directory.  For training on WildCross, the three config files for the above network architectures respectively are `exp_wildvpr_range_res50.py`, `exp_wildvpr_range_dinov2.py` and `exp_wildvpr_range_dinov3.py`.  Before training, first change the value of  `data_path` in the relevant config file to point towards the root directory of the *WildCross* dataset on your machine.
### DinoV3
In order to train DinoV3, first follow the instructions on [The official DinoV3 repository](https://github.com/facebookresearch/dinov3) to clone the repository to your local machine and download the pre-trained weights.  Then, add the paths to the repository and pretrained weights where indicated in `LIP-Loc/models/CLIPModelV1NormWildCross.py` in order to train with this model architecture.

