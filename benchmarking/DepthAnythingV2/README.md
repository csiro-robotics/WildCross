# Metric Depth Estimation Benchmarking and Evaluation

This subfolder contains scripts for training and evaluation for the task of Metric Depth Estimation for **WildCross**, using DepthAnythingV2.

## Setup

### Dataset 
To download the **WildCross** Dataset, follow the instructions in the root directory of this repository.  By default this repository will use the full sized images (`images`) and point clouds (`depth`) for training and evaluation.

### Environment
We provide an environment file to set up the necessary python environment for training and evaluation using **mamba**.  The environment can be installed by running the following command out of this directory:

```
mamba install -f environment.yaml
```

### Docker
We provide a dockerfile that will enable building the base docker image that can be used to install any of the benchmarking environments (`cross_modal`, `DepthAnythingV2`, and `VPR`).

The environments themselves need gpus to compile and so we recommend the following process:

1. Build the base docker image
```
docker build -t wildcross:unbuilt -f docker/Dockerfile .
```
2. Begin an interactive container with access to GPUs
```
docker run -it --rm --gpus all --name wildcross wildcross:unbuilt
```
3. Within interactive container, build conda envrionments
```
mamba env create -f /workspace/VPR/environment.yaml -y && \
mamba env create -f /workspace/DepthAnythingV2/environment.yaml -y && \
mamba env create -f /workspace/cross_modal/environment.yaml -y && \
mamba clean --all
```
4. Keeping the interactive container window open, in a fresh terminal, save the built image
```
docker container commit wildcross wildcross:latest
```

With final docker container, you can activate environments and run scripts for each of the benchmarking scenarios.

### Checkpoints 
We provide download links for both the KITTI pre-trained and WildCross fine-tuned checkpoints for each network backbone:

| Backbone | Checkpoint | Link |
|:-|:-|:-:|
|ViT-s|Pre-trained|[Download]()|
||Fine-tuned|[Download]()|
|ViT-b|Pre-trained|[Download]()|
||Fine-tuned|[Download]()|
|ViT-l|Pre-trained|[Download]()|
||Fine-tuned|[Download]()|

## Training
To train on the WildCross dataset, firstly edit the value of the `wildcross_root` variable on line 74 of `train.py` to the directory containing the WildCross dataset on your machine.  Then, you can train the network by running `train.py` as follows:

```
export PYTHONPATH=$PWD:$PYTHONPATH 
GPUS=$NUM_GPUS

python3 -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20596 \
    train.py \
        --encoder $ENCODER \
        --save-path /path/to/save/directory \
        --pretrained-from /path/to/pretrained/checkpoint \
```

Where:
- `GPUS` refers to the number of GPUS to be used for training
- `--encoder` refers to the backbone encoder to be used.  We recommend using `vits`, `vitb` or `vitl` for small, base or large Visual Transformers respectively.
- `--save-path` refers to the directory to save training logs and checkpoints in 
- `--pretrained-from` refers to the checkpoint to use as the initialisation for training.  We strongly recommend using the KITTI pre-trained model to intialise the network before fine-tuning on WildCross.

## Evaluation
To evaluate a given checkpoint, run `eval.py` as follows:

```
export PYTHONPATH=$PWD:$PYTHONPATH
python3 eval.py \
    --encoder $ENCODER \
    --checkpoint /path/to/checkpoint.pth \
```
Where `--encoder` is the same as used in the training script, and here `--checkpoint` refers to the checkpoint being evaluated.


## Acknowledgements
We would like again to acknowledge the authors of DepthAnythingV2 and the maintainers of the open source implementation found at [https://github.com/DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2), which has formed the basis for the code in this repository.