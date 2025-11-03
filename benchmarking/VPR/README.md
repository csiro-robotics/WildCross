# Visual Place Recognition: Benchmarking and Evaluation

This subfolder contains scripts for training and evaluation the task of Visual Place Recognition for **WildCross**.  We provide support for the following VPR methods:

| Method | Paper | GitHub |
|:-|:-:|:-:|
| NetVLAD | [Link](https://openaccess.thecvf.com/content_cvpr_2016/papers/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.pdf) | [Source](https://github.com/gmberton/VPR-methods-evaluation) |
| MixVPR | [Link](https://openaccess.thecvf.com/content/WACV2023/papers/Ali-bey_MixVPR_Feature_Mixing_for_Visual_Place_Recognition_WACV_2023_paper.pdf) | [Source](https://github.com/amaralibey/MixVPR) |
| SALAD | [Link](https://openaccess.thecvf.com/content/CVPR2024/papers/Izquierdo_Optimal_Transport_Aggregation_for_Visual_Place_Recognition_CVPR_2024_paper.pdf) | [Source](https://github.com/serizba/salad) |
| BoQ | [Link](https://openaccess.thecvf.com/content/CVPR2024/papers/Ali-bey_BoQ_A_Place_is_Worth_a_Bag_of_Learnable_Queries_CVPR_2024_paper.pdf) | [Source](https://github.com/amaralibey/Bag-of-Queries) |

## Setup 

### Dataset 
To download the **WildCross** Dataset, follow the instructions in the root directory of this repository.  By default this repository will use the pre-shrunk images (*i.e.* images in the `images_shrunk` folder) to optimise the speed at which the raw data is loaded from the disk, but this can be changed by editing `./data/dataset.py`.

### Environment
We provide an environment file to set up the necessary python environment for training and evaluation using **mamba**.  The environment can be installed by running the following command out of this directory:

```
mamba install -f environment.yaml
```

### Checkpoints
We provide download links for both the author-provided pre-trained checkpoints on urban environments and our fine-tuned checkpoints for each method:

| Method | Checkpoint | Link |
|:-|:-|:-:|
|NetVlad|Pre-Trained|[Download]()|
||Split 1|[Download]()|
||Split 2|[Download]()|
||Split 3|[Download]()|
||Split 4|[Download]()|
|MixVPR|Pre-Trained|[Download]()|
||Split 1|[Download]()|
||Split 2|[Download]()|
||Split 3|[Download]()|
||Split 4|[Download]()|
|SALAD|Pre-Trained|[Download]()|
||Split 1|[Download]()|
||Split 2|[Download]()|
||Split 3|[Download]()|
||Split 4|[Download]()|
|BoQ|Pre-Trained|[Download]()|
||Split 1|[Download]()|
||Split 2|[Download]()|
||Split 3|[Download]()|
||Split 4|[Download]()|

## Training
To train a given method on the WildCross Dataset, run `train.py` as follows:
```
export PYTHONPATH=$PWD:$PYTHONPATH
python train.py \
    --config /path/to/method/configs.yaml \
    --save_dir /path/to/save/dir \
    --pretrained_ckpt /path/to/checkpoint.pth \
    --split_idx 
```

Where:
- `--config` is for providing the config file for a given method, which can be found in `./configs`
- `--savedir` is for providing the directory to save checkpoints and logs in
- `--pretrained_ckpt` is for providing a path to a pre-trained checkpoint to load before starting training.  For best performance, we recommend using the urban pre-trained checkpoints provided here.

It is also possible to modify arguments in the config file through the command line by passing the name of the argument and its value as a pair of inputs at the end of the command, (*e.g.* `max_epochs 20` to modify the maximum number of training epochs)



## Evaluation
For evaluating a trained model, run `eval_inter_sequence.py` or `eval_intra_sequence.py` for inter or intra-sequence evaluation respectively as follows:

```
export PYTHONPATH=$PWD:$PYTHONPATH

```