# W-HMR: Human Mesh Recovery in World Space with Weak-supervised Camera Calibration and Orientation Correction
[![report](https://img.shields.io/badge/Project-Page-blue)](https://yw0208.github.io/w-hmr/)
[![report](https://img.shields.io/badge/ArXiv-Paper-red)](https://arxiv.org/abs/2311.17460v3)

<p float="center">
  <img src="docs/demo.png" width="100%" />
</p>

## Features

W-HMR is a human body pose and shape estimation method in world space. 
It predicts the SMPL body model in both camera and world coordinates for a monocular image.
[Camera Calibration] predicts the focal length for better mesh-image alignment. 
[Orientation Correction] make recovered body reasonable in world space.

This implementation:
- has the demo code for W-HMR implemented in PyTorch.

## News 🚩

[March 21, 2024] Release codes and pretrained weights for demo.

[March 26, 2024] Pre-processed labels are available now.

## TODOs

- [x] Release demo codes.

- [x] Release pre-processed labels. 

- [ ] Release evluation codes.  

- [ ] Release training codes.  

## Getting Started
### Requirements
W-HMR has been implemented and tested on Ubuntu 18.04 with python == 3.8.

Install the requirements following environment.yml

### Pre-processed Dataset Labels
All the data used in our paper is publicly available. You can just download them from their official website following our
dataset introduction in the [paper](https://arxiv.org/abs/2311.17460v3).

But for your convenience, I provide some download links for pre-processed labels here.

The most important source is [PyMAF](https://github.com/HongwenZhang/PyMAF). You can download the pre-processed labels of 
3DPW, COCO, LSP, MPII and MPI-INF-3DHP, which include pseudo 3D joint label fitted by EFT.

We also use some augmented data from [CLIFF](https://github.com/huawei-noah/noah-research) and [HMR 2.0](https://github.com/shubham-goel/4D-Humans).

I also processed some dataset labels (e.g. AGORA and HuMMan), you can download them from [here](https://drive.google.com/file/d/1-R9Spqb3MG5b5FNQTrf8iH2vWMKwovAX/view?usp=sharing). 






## Running the Demo

### W-HMR

 First, you need to download the required data 
(i.e our trained model and SMPL model parameters) from [here](https://drive.google.com/file/d/1zdQ3nPRgoHr7fM_U-6olxA9qgFHytaoe/view?usp=sharing). It is approximately 700MB.
Unzip it and put it in the repository root. Then, running the demo is as simple as:

```shell
python demo/whmr_demo.py --image_folder data/sample_images --output_folder output/sample_images
```

Sample demo output:

<p float="left">
  <img src="docs/demo_result.png" width="99%" />
</p>

On the right, they are the output in camera and world coordinate. We put them in world space and generate ground planes to show how accurate the orientation is.

## Training

Training instructions will follow after publication.

## Acknowledgments

Part of the code is borrowed from the following projects, including ,[PyMAF](https://github.com/HongwenZhang/PyMAF), [AGORA](https://github.com/pixelite1201/agora_evaluation), [PyMAF-X](https://github.com/HongwenZhang/PyMAF-X), [PARE](https://github.com/mkocabas/PARE), [SPEC](https://github.com/mkocabas/SPEC), [MeshGraphormer](https://github.com/microsoft/MeshGraphormer), [4D-Humans](https://github.com/shubham-goel/4D-Humans), [VitPose](https://github.com/ViTAE-Transformer/ViTPose). Many thanks to their contributions.
