# BoXHED2.0

What’s new (over BoXHED1.0):
 - Allows for survival data beyond right censoring, including recurrent events
 - Significant speed improvement
 - Multicore CPU and GPU support


Please refer to [BoXHED2.0 paper](https://arxiv.org/abs/2103.12591) for details, which builds on [BoXHED1.0 paper](http://proceedings.mlr.press/v119/wang20o/wang20o.pdf) (ICML 2020). The theoretical underpinnings for BoXHED is provided [here](https://projecteuclid.org/journals/annals-of-statistics/volume-49/issue-4/Boosted-nonparametric-hazards-with-time-dependent-covariates/10.1214/20-AOS2028.full) (Annals of Statistics 2021).

## Prerequisites
The software was developed and tested in Linux and Mac OS environments. The requirements are the following:
- cmake  (>=3.18.2) (Windows users: CMake is part of the Visual Studio installation)
- Python (=3.8)
- [CUDA](https://developer.nvidia.com/cuda-11.1.1-download-archive)   (=11.1) (only if GPU support is needed)
- conda  (we recommend using the free [Anaconda distribution](https://docs.anaconda.com/anaconda/install/))

Windows users need to have the Visual Studio 17 2022 toolset installed. During installation, under the "Workloads" tab select "Desktop Development with C++" in the "Desktop and Mobile" section. Make the following selections in the menu that shows up on the right:
![sc__](https://user-images.githubusercontent.com/34462617/201495851-c7d02796-31e0-4181-9eba-78065d2a5f59.png)

## [Installing BoXHED2.0 with multicore CPU support](README_CPU.md)

## [Installing BoXHED2.0 with GPU support](README_GPU.md)