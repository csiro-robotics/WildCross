# Cross-Modal Benchmarking and Evaluation

This subfolder contains scripts for training and evaluating models for the task of Cross-Modal Place Recognition using the **WildCross** dataset.  We provide support for the baseline cross-modal approach LIP-Loc, using the ResNet50, DinoV2 and DinoV3 backbones.  

## Setup

### Dataset 
To download the **WildCross** Dataset, follow the instructions in the root directory of this repository.  By default this repository will use the full resolution point clouds (`Clouds`) and the pre-shrunk images (*i.e.* images in the `images_shrunk` folder) to optimise the speed at which the raw data is loaded from the disk, but this can be changed by editing `./dataloaders/WildCrossDatasetRangeView.py`.

### Environment
We provide a conda environment file to set up the necessary python environment for training and evaluation. For best results we recommend using **mamba** over the original conda.  The environment can be installed by running the following command out of this directory:

```
mamba env create -f environment.yaml
```

### DINOv3
To train with DINOv3, at time of publication, you need to clone the official repo and checkpoint path yourself.
See https://github.com/facebookresearch/dinov3 for details on the repo and the checkpoints. Our model uses the vits16 pre-trained dinov3 model.
Once these are downloaded, you need to define `REPO_DIR` and `CHECKPOINT_PATH` within line 57 and 58 of `benchmarking/cross_modal/models/CLIPModelV1NormWildCross.py`. Current default lines are shown below for convenience.

```
REPO_DIR="/path/to/dinov3"
CHECKPOINT_PATH="/path/to/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
```

### Checkpoints 
We provide download links for our fine-tuned checkpoints for each network backbone:

| Backbone | Split | Link |
|:-|:-|:-:|
|ResNet50|Split 1|[Download](https://huggingface.co/CSIRORobotics/WildCross/resolve/main/crossmodal/resnet50/split_0.pth?download=true)|
||Split 2|[Download](https://huggingface.co/CSIRORobotics/WildCross/resolve/main/crossmodal/resnet50/split_1.pth?download=true)|
||Split 3|[Download](https://huggingface.co/CSIRORobotics/WildCross/resolve/main/crossmodal/resnet50/split_2.pth?download=true)|
||Split 4|[Download](https://huggingface.co/CSIRORobotics/WildCross/resolve/main/crossmodal/resnet50/split_3.pth?download=true)|
|DinoV2|Split 1|[Download](https://huggingface.co/CSIRORobotics/WildCross/resolve/main/crossmodal/dinov2/split_0.pth?download=true)|
||Split 2|[Download](https://huggingface.co/CSIRORobotics/WildCross/resolve/main/crossmodal/dinov2/split_1.pth?download=true)|
||Split 3|[Download](https://huggingface.co/CSIRORobotics/WildCross/resolve/main/crossmodal/dinov2/split_2.pth?download=true)|
||Split 4|[Download](https://huggingface.co/CSIRORobotics/WildCross/resolve/main/crossmodal/dinov2/split_3.pth?download=true)|
|DinoV3|Split 1|[TBA]()|
||Split 2|[TBA]()|
||Split 3|[TBA]()|
||Split 4|[TBA]()|

## Training
To train a given backbone and split on the WildCross Dataset, run `trainer_wildcross.py` as follows:

```
export PYTHONPATH=$PWD:$PYTHONPATH 
python trainer_wildcross.py \
    --expid $EXPID \
    --save_dir /path/to/save_dir \
    --split_idx $SPLIT_IDX
```

Where:
- `--expid` is the name of the config file for this experiment.  Available config files can be found in `configs`, with the relevant examples for training on WildCross being:
    - `exp_wildcross_range_resnet50`: Training with ResNet50 
    - `exp_wildcross_range_dinov2`: Training with DinoV2 
    - `exp_wildcross_range_dinov3`: Training with DinoV3 
- `--save_dir` is for the directory which the training logs and checkpoints will be saved to
- `--split_idx` is for indicating which of the four crossfold training splits is being trained in this experiment.  **Note**: In this codebase, the crossfold dataset splits are 0-indexed (*e.g.* split 0, 1, 2, 3) unlike the paper and links above that are 1-indexed - keep this in mind when selecting splits for training and testing.

Final model will be saved in `save_dir` under `latest.pth`

## Evaluation
To evaluate a given checkpoint, run `evaluate_wildcross/evaluate.py` as follows:

```
export PYTHONPATH=$PWD:$PYTHONPATH
python evaluate_wildcross/evaluate.py \
    --expid $EXPID \
    --save_dir /path/to/save_dir \
    --split_idx $SPLIT_IDX
    --ckpt /path/to/checkpoint.pth \
```

Where:
- `--expid`, `--save_dir` and `--split_idx` are the same as for the training script
- `--ckpt` is for setting the path to the checkpoint to be evaluated 

**Note** to attain the results shown in the paper (Table VIII), training and testing must be done across all splits and then averaged. There is no script currently for automating this process.

## Acknowledgements
We would like again to acknowledge the authors of the original LIP-Loc paper and the maintainers of the open source repository hosted at [https://github.com/Shubodh/lidar-image-pretrain-VPR](https://github.com/Shubodh/lidar-image-pretrain-VPR) which is used as the basis for the code in this repository