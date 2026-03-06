<div align="center">
<h1>WildCross: A Cross-Modal Large Scale Benchmark for Place Recognition and Metric Depth Estimation in Natural Environments</h1>

<h2> Accepted at IEEE ICRA 2026 </h2>

[**Joshua Knights**](https://scholar.google.com/citations?user=RxbGr2EAAAAJ&hl=en)<sup>1,2</sup> , **Joseph Reid**<sup>1</sup> , [**Kaushik Roy**](https://bit0123.github.io/)<sup>1</sup> , [**David Hall**](https://scholar.google.com/citations?user=dosODoQAAAAJ&hl=en)<sup>1</sup> , [**Mark Cox**](https://scholar.google.com/citations?user=Bk3UD4EAAAAJ&hl=en)<sup>1</sup> , [**Peyman Moghadam**](https://scholar.google.com.au/citations?user=QAVcuWUAAAAJ&hl=en)<sup>1,2</sup>

<sup>1</sup>CSIRO Robotics&emsp;&emsp;&emsp;<sup>2</sup>Queensland University of Technology
<br>

<a href="https://arxiv.org/abs/2603.01475"><img src='https://img.shields.io/badge/arXiv-WildCross-red' alt='Paper PDF'></a>
<a href='https://csiro-robotics.github.io/WildCross'><img src='https://img.shields.io/badge/Project_Page-WildCross-green' alt='Project Page'></a>
<a href='https://doi.org/10.25919/5fmy-yg37'><img src='https://img.shields.io/badge/Dataset_Download-WildCross-blue'></a>
</div>

This repository contains supporting scripts for downloading, training and evaluating techniques on  **WildCross**, a large-scale multi-modal benchmark for place recognition and metric depth estimation in natural environments accepted at IEEE ICRA2026.

![teaser](media/teaser.png)


## News

- **FEBRUARY 2026:** Paper, project page, code, models, demo, and benchmark are all released.
- **JANUARY 2026:** WildCross Paper is accepted to IEEE ICRA 2026. 

## Download Instructions
Our dataset can be downloaded through the [**CSIRO Data Access Portal**]('https://doi.org/10.25919/5fmy-yg37').  Detailed instructions for downloading the dataset can be found in the README file provided on the data access portal page.

## Usage

### Data loading and visualisation scripts
Instructions on how to load each of the data modalities (RGB Image, Depth Image, Surface Normal Image, 3D Submap) can be found in `example_loaders.py`, and a script to visualise the depth image using the palette employed in our visualisations can be found in `visualise_depth_images.py`

### Training and Evaluation
We provide code for training and evaluation on the **WildCross** dataset for the tasks of [visual place recognition (VPR)](benchmarking/VPR/), [cross-modal place recognition (CMPR)](/benchmarking/cross_modal/) and [metric depth estimation](benchmarking/DepthAnythingV2/), which can be found in their respectives subfolders inside the `benchmarking` folder.  For more detailed instructions for setting up and running these benchmarks, consult the documentation inside the respective subfolders for each task. For [LiDAR place recognition (LPR)](benchmarking/LPR/) code for training and evaluation can be found on a new WildCross_subsets branch of the original [Wild-Places](https://github.com/csiro-robotics/Wild-Places) repository. Information for getting to the branch can be found inside the `benchmarking/LPR` folder.

### Checkpoints
We provide checkpoints for the fine-tuned models used to produce the results in this publication, as well as the urban pre-trained models provided by the original authors for each of the benchmarked methods where applicable.  See the sub-folder for each benchmarked task to find the respective download links to the relevant checkpoints for that task. You can also find all weights on the WildCross [HuggingFace page](https://huggingface.co/CSIRORobotics/WildCross)

## Acknowledgements
We are grateful to the authors and open source maintainers for NetVlad, [MixVPR](https://github.com/amaralibey/MixVPR), [SALAD](https://github.com/serizba/salad), [BOQ](https://github.com/amaralibey/Bag-of-Queries), [LIP-Loc](https://github.com/Shubodh/lidar-image-pretrain-VPR) and [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2), whose implementations form the basis of the training and evaluation code presented in this repository.  We would also like to thank the author of the [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation) repository, which provided an excellent starting point for the development of the evaluation scripts in this repository.  

## Citation
If you find this repository useful or use the WildCross dataset in your work, please cite the paper using the following:
```
@inproceedings{wildcross2026,
  title={{WildCross: A Cross-Modal Large Scale Benchmark for Place Recognition and Metric Depth Estimation in Natural Environments}},
  author={Joshua Knights, Joseph Reid, Kaushik Roy, David Hall, Mark Cox, Peyman Moghadam},
  booktitle={Proceedings-IEEE International Conference on Robotics and Automation},
  pages={},
  year={2026}
}
```
And the original WildPlaces paper:

```
@inproceedings{knights2023wildplaces,
  title={{Wild-Places: A Large-Scale Dataset for Lidar Place Recognition in Unstructured Natural Environments},
  author={Knights, Joshua and Vidanapathirana, Kavisha and Ramezani, Milad and Sridharan, Sridha and Fookes, Clinton and Moghadam, Peyman},
  booktitle={Proceedings-IEEE International Conference on Robotics and Automation},
  pages={11322--11328},
  year={2023}
}
```

