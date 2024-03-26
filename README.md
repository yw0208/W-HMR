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

## TODOs

- [x] Release demo codes.

- [ ] Release pre-processed labels. 

- [ ] Release evluation codes.  

- [ ] Release training codes.  

## Getting Started

W-HMR has been implemented and tested on Ubuntu 18.04 with python == 3.8.

Install the requirements following environment.yml

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
