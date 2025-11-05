<div align="center">
<h1>WildCross: A Cross-Modal Large Scale Benchmark for Place Recognition and Metric Depth Estimation in Natural Environments</h1>

[**Joshua Knights**](https://scholar.google.com/citations?user=RxbGr2EAAAAJ&hl=en)<sup>1</sup> · **Joseph Reid**<sup>1,2</sup> · [**Mark Cox**](https://scholar.google.com/citations?user=Bk3UD4EAAAAJ&hl=en)<sup>1</sup>
<br>
[**Kaushik Roy**](https://bit0123.github.io/)<sup>1</sup> · [**David Hall**](https://scholar.google.com/citations?user=dosODoQAAAAJ&hl=en)<sup>1</sup> · [**Peyman Moghadam**](https://scholar.google.com.au/citations?user=QAVcuWUAAAAJ&hl=en)<sup>1</sup>

<sup>1</sup>DATA61, CSIRO&emsp;&emsp;&emsp;<sup>2</sup>University of Queensland
<br>

<a href="/link/to/arxiv"><img src='https://img.shields.io/badge/arXiv-WildCross-red' alt='Paper PDF'></a>
<a href='https://depth-anything-v2.github.io'><img src='https://img.shields.io/badge/Project_Page-WildCross-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/depth-anything/Depth-Anything-V2'><img src='https://img.shields.io/badge/Dataset_Download-WildCross-blue'></a>
</div>

This repository contains supporting scripts for downloading, training and evaluation for **WildCross**, a large-scale multi-modal benchmark for place recognition and metric depth estimation in natural environments.

![teaser](media/teaser.png)


## News

- **20XX-XX-XX:** Paper, project page, code, models, demo, and benchmark are all released.

## Download Instructions
Our dataset can be downloaded through the [**CSIRO Data Access Portal**]('/link/to/portal/when/up').  Detailed instructions for downloading the dataset can be found in the README file provided on the data access portal page.

## Usage

### Data loading and visualisation scripts
Instructions on how to load each of the data modalities (RGB Image, Depth Image, 3D Submap) can be found in `example_loaders.py`, and a script to visualise the depth image using the palette employed in our visualisations can be found in `visualise_depth_images.py`

### Training and Evaluation
We provide code for training and evaluation on the **WildCross** dataset for the tasks of [visual place recognition (VPR)](benchmarking/VPR/), [cross-modal place recognition (CMPR)](/benchmarking/cross_modal/) and [metric depth estimation](benchmarking/DepthAnythingV2/), which can be found in their respectives subfolders inside the `benchmarking` folder.  For more detailed instructions for setting up and running these benchmarks, consult the documentation inside the respective subfolders for each task.

### Checkpoints
We provide checkpoints for the fine-tuned models used to produce the results in this publication, as well as the urban pre-trained models provided by the original authors for each of the benchmarked methods where applicable.  See the sub-folder for each benchmarked task to find the respective download links to the relevant checkpoints for that task.

## Acknowledgements
We are grateful to the authors and open source maintainers for NetVlad, [MixVPR](https://github.com/amaralibey/MixVPR), [SALAD](https://github.com/serizba/salad), [BOQ](https://github.com/amaralibey/Bag-of-Queries), [LIP-Loc](https://github.com/Shubodh/lidar-image-pretrain-VPR) and [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2), whose implementations form the basis of the training and evaluation code presented in this repository.  We would also like to thank the author of the [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation) repository, which provided an excellent starting point for the development of the evaluation scripts in this repository.  

## Citation
If you find this repository useful or use the WildCross dataset in your work, please cite us using the following:
```
@misc{knights2025wildcross,
  title={{WildCross: A Cross-Modal Large Scale Benchmark for Place Recognition and Metric Depth Estimation in Natural Environments}},
  author={Joshua Knights, Joseph Reid, Mark Cox, Kaushik Roy, David Hall, Peyman Moghadam},
  year={2025},
  eprint={xxxxxxxxx},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/xxxxxxxxxx},
}
```

<!-- 
## Pre-trained Models

We provide **four models** of varying scales for robust relative depth estimation:

| Model | Params | Checkpoint |
|:-|-:|:-:|
| Depth-Anything-V2-Small | 24.8M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true) |
| Depth-Anything-V2-Base | 97.5M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true) |
| Depth-Anything-V2-Large | 335.3M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) |
| Depth-Anything-V2-Giant | 1.3B | Coming soon |


## Usage

### Prepraration

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
pip install -r requirements.txt
```

Download the checkpoints listed [here](#pre-trained-models) and put them under the `checkpoints` directory.

### Use our models
```python
import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('your/image/path')
depth = model.infer_image(raw_img) # HxW raw depth map in numpy
```

If you do not want to clone this repository, you can also load our models through [Transformers](https://github.com/huggingface/transformers/). Below is a simple code snippet. Please refer to the [official page](https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything_v2) for more details.

- Note 1: Make sure you can connect to Hugging Face and have installed the latest Transformers.
- Note 2: Due to the [upsampling difference](https://github.com/huggingface/transformers/pull/31522#issuecomment-2184123463) between OpenCV (we used) and Pillow (HF used), predictions may differ slightly. So you are more recommended to use our models through the way introduced above.
```python
from transformers import pipeline
from PIL import Image

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
image = Image.open('your/image/path')
depth = pipe(image)["depth"]
```

### Running script on *images*

```bash
python run.py \
  --encoder <vits | vitb | vitl | vitg> \
  --img-path <path> --outdir <outdir> \
  [--input-size <size>] [--pred-only] [--grayscale]
```
Options:
- `--img-path`: You can either 1) point it to an image directory storing all interested images, 2) point it to a single image, or 3) point it to a text file storing all image paths.
- `--input-size` (optional): By default, we use input size `518` for model inference. ***You can increase the size for even more fine-grained results.***
- `--pred-only` (optional): Only save the predicted depth map, without raw image.
- `--grayscale` (optional): Save the grayscale depth map, without applying color palette.

For example:
```bash
python run.py --encoder vitl --img-path assets/examples --outdir depth_vis
```

### Running script on *videos*

```bash
python run_video.py \
  --encoder <vits | vitb | vitl | vitg> \
  --video-path assets/examples_video --outdir video_depth_vis \
  [--input-size <size>] [--pred-only] [--grayscale]
```

***Our larger model has better temporal consistency on videos.***

### Gradio demo

To use our gradio demo locally:

```bash
python app.py
```

You can also try our [online demo](https://huggingface.co/spaces/Depth-Anything/Depth-Anything-V2).

***Note: Compared to V1, we have made a minor modification to the DINOv2-DPT architecture (originating from this [issue](https://github.com/LiheYoung/Depth-Anything/issues/81)).*** In V1, we *unintentionally* used features from the last four layers of DINOv2 for decoding. In V2, we use [intermediate features](https://github.com/DepthAnything/Depth-Anything-V2/blob/2cbc36a8ce2cec41d38ee51153f112e87c8e42d8/depth_anything_v2/dpt.py#L164-L169) instead. Although this modification did not improve details or accuracy, we decided to follow this common practice.


## Fine-tuned to Metric Depth Estimation

Please refer to [metric depth estimation](./metric_depth).


## DA-2K Evaluation Benchmark

Please refer to [DA-2K benchmark](./DA-2K.md).


## Community Support

**We sincerely appreciate all the community support for our Depth Anything series. Thank you a lot!**

- Apple Core ML:
    - https://developer.apple.com/machine-learning/models
    - https://huggingface.co/apple/coreml-depth-anything-v2-small
    - https://huggingface.co/apple/coreml-depth-anything-small
- Transformers:
    - https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything_v2
    - https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything
- TensorRT:
    - https://github.com/spacewalk01/depth-anything-tensorrt
    - https://github.com/zhujiajian98/Depth-Anythingv2-TensorRT-python
- ONNX: https://github.com/fabio-sim/Depth-Anything-ONNX
- ComfyUI: https://github.com/kijai/ComfyUI-DepthAnythingV2
- Transformers.js (real-time depth in web): https://huggingface.co/spaces/Xenova/webgpu-realtime-depth-estimation
- Android:
  - https://github.com/shubham0204/Depth-Anything-Android
  - https://github.com/FeiGeChuanShu/ncnn-android-depth_anything


## Acknowledgement

We are sincerely grateful to the awesome Hugging Face team ([@Pedro Cuenca](https://huggingface.co/pcuenq), [@Niels Rogge](https://huggingface.co/nielsr), [@Merve Noyan](https://huggingface.co/merve), [@Amy Roberts](https://huggingface.co/amyeroberts), et al.) for their huge efforts in supporting our models in Transformers and Apple Core ML.

We also thank the [DINOv2](https://github.com/facebookresearch/dinov2) team for contributing such impressive models to our community.


## LICENSE

Depth-Anything-V2-Small model is under the Apache-2.0 license. Depth-Anything-V2-Base/Large/Giant models are under the CC-BY-NC-4.0 license.


## Citation

If you find this project useful, please consider citing:

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
``` -->